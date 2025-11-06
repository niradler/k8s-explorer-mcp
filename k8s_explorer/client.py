"""Kubernetes client wrapper with caching and permission awareness."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .cache import K8sCache
from .fuzzy_matching import FuzzyResourceMatcher
from .models import CacheKey, ResourceIdentifier
from .permissions import PermissionChecker

logger = logging.getLogger(__name__)


class K8sClient:
    """Kubernetes client with caching and permission awareness."""

    def __init__(self, cache: Optional[K8sCache] = None, check_permissions: bool = True):
        """Initialize K8s client."""
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded Kubernetes config from kubeconfig")
            except config.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise

        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.api_client = client.ApiClient()

        self.cache = cache or K8sCache()
        self.permission_checker = PermissionChecker(self.api_client) if check_permissions else None
        self._api_mapping = self._build_api_mapping()

        if self.permission_checker:
            logger.info("Permission checking enabled")

    def _build_api_mapping(self) -> Dict[str, Any]:
        """Build mapping of resource kinds to API methods."""
        return {
            "Pod": {
                "api": self.core_v1,
                "list_namespaced": "list_namespaced_pod",
                "read_namespaced": "read_namespaced_pod",
                "list_all": "list_pod_for_all_namespaces",
            },
            "Service": {
                "api": self.core_v1,
                "list_namespaced": "list_namespaced_service",
                "read_namespaced": "read_namespaced_service",
                "list_all": "list_service_for_all_namespaces",
            },
            "Deployment": {
                "api": self.apps_v1,
                "list_namespaced": "list_namespaced_deployment",
                "read_namespaced": "read_namespaced_deployment",
                "list_all": "list_deployment_for_all_namespaces",
            },
            "ReplicaSet": {
                "api": self.apps_v1,
                "list_namespaced": "list_namespaced_replica_set",
                "read_namespaced": "read_namespaced_replica_set",
                "list_all": "list_replica_set_for_all_namespaces",
            },
            "StatefulSet": {
                "api": self.apps_v1,
                "list_namespaced": "list_namespaced_stateful_set",
                "read_namespaced": "read_namespaced_stateful_set",
                "list_all": "list_stateful_set_for_all_namespaces",
            },
            "DaemonSet": {
                "api": self.apps_v1,
                "list_namespaced": "list_namespaced_daemon_set",
                "read_namespaced": "read_namespaced_daemon_set",
                "list_all": "list_daemon_set_for_all_namespaces",
            },
            "ControllerRevision": {
                "api": self.apps_v1,
                "list_namespaced": "list_namespaced_controller_revision",
                "read_namespaced": "read_namespaced_controller_revision",
                "list_all": "list_controller_revision_for_all_namespaces",
            },
            "ConfigMap": {
                "api": self.core_v1,
                "list_namespaced": "list_namespaced_config_map",
                "read_namespaced": "read_namespaced_config_map",
                "list_all": "list_config_map_for_all_namespaces",
            },
            "Secret": {
                "api": self.core_v1,
                "list_namespaced": "list_namespaced_secret",
                "read_namespaced": "read_namespaced_secret",
                "list_all": "list_secret_for_all_namespaces",
            },
        }

    async def get_resource(
        self, resource_id: ResourceIdentifier, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get a resource by identifier."""
        if use_cache:
            cached = self.cache.get_resource(resource_id)
            if cached is not None:
                return cached

        try:
            if resource_id.kind in self._api_mapping:
                mapping = self._api_mapping[resource_id.kind]
                api = mapping["api"]
                method_name = mapping["read_namespaced"]
                method = getattr(api, method_name)

                if resource_id.namespace:
                    result = method(resource_id.name, resource_id.namespace)
                else:
                    result = method(resource_id.name)

                resource_dict = self.api_client.sanitize_for_serialization(result)

                if use_cache:
                    self.cache.set_resource(resource_id, resource_dict)

                return resource_dict

        except ApiException as e:
            if e.status == 404:
                logger.debug(f"Resource not found: {resource_id.kind}/{resource_id.name}")
                return None
            logger.error(f"Error fetching resource: {e}")
            raise

    async def get_resource_or_similar(
        self,
        resource_id: ResourceIdentifier,
        use_cache: bool = True,
        enable_fuzzy_match: bool = True,
        similarity_threshold: float = 0.7,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Get a resource by identifier, with fuzzy matching fallback.

        Args:
            resource_id: Resource identifier
            use_cache: Whether to use cache
            enable_fuzzy_match: Enable fuzzy matching if exact match fails
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            Tuple of (resource, match_info)
            - resource: The found resource or None
            - match_info: Dict with match details if fuzzy match was used, None if exact match
        """
        resource = await self.get_resource(resource_id, use_cache=use_cache)

        if resource is not None:
            return resource, None

        if not enable_fuzzy_match or resource_id.kind not in ["Pod"]:
            return None, None

        logger.info(
            f"Exact match not found for {resource_id.kind}/{resource_id.name}, trying fuzzy match..."
        )

        try:
            available_resources, _ = await self.list_resources(
                kind=resource_id.kind,
                namespace=resource_id.namespace,
                use_cache=use_cache,
                check_permissions=False,
            )

            if not available_resources:
                return None, None

            match_result = FuzzyResourceMatcher.find_best_match(
                resource_id.name, available_resources, threshold=similarity_threshold
            )

            if match_result:
                matched_resource, score, reason = match_result
                matched_name = matched_resource.get("metadata", {}).get("name", "")

                match_info = {
                    "fuzzy_match_used": True,
                    "original_name": resource_id.name,
                    "matched_name": matched_name,
                    "similarity_score": score,
                    "match_reason": reason,
                    "explanation": FuzzyResourceMatcher.explain_match(
                        resource_id.name, matched_name, score, reason
                    ),
                }

                logger.info(
                    f"Fuzzy match found: {matched_name} (score: {score:.2f}, reason: {reason})"
                )
                return matched_resource, match_info

            logger.info(f"No similar {resource_id.kind} found for {resource_id.name}")
            return None, None

        except Exception as e:
            logger.error(f"Error during fuzzy matching: {e}")
            return None, None

    async def list_resources(
        self,
        kind: str,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
        use_cache: bool = True,
        check_permissions: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict]]:
        """
        List resources of a kind.

        Returns:
            Tuple of (resources, permission_notice)
            permission_notice is None if full access, otherwise contains details
        """
        permission_notice = None

        if check_permissions and self.permission_checker:
            checks = await self.permission_checker.check_resource_access(kind, namespace)

            list_check = checks.get("list")
            if list_check and not list_check.allowed:
                logger.warning(f"No permission to list {kind} in {namespace or 'cluster'}")
                permission_notice = self.permission_checker.create_permission_notice(
                    kind, namespace, checks
                )
                return [], permission_notice

        cache_key = CacheKey(namespace=namespace, kind=kind, label_selector=label_selector)

        if use_cache:
            cached = self.cache.get_list_query(cache_key)
            if cached is not None:
                return cached, permission_notice

        try:
            if kind in self._api_mapping:
                mapping = self._api_mapping[kind]
                api = mapping["api"]

                if namespace:
                    method_name = mapping["list_namespaced"]
                    method = getattr(api, method_name)
                    result = method(namespace, label_selector=label_selector)
                else:
                    method_name = mapping["list_all"]
                    method = getattr(api, method_name)
                    result = method(label_selector=label_selector)

                resources = []
                for item in result.items:
                    resource = self.api_client.sanitize_for_serialization(item)
                    # K8s list items don't include kind/apiVersion, so add them manually
                    if "kind" not in resource:
                        resource["kind"] = kind
                    if "apiVersion" not in resource and hasattr(item, "api_version"):
                        resource["apiVersion"] = item.api_version
                    resources.append(resource)

                if use_cache:
                    self.cache.set_list_query(cache_key, resources)

                return resources, permission_notice

        except ApiException as e:
            if e.status == 403:
                logger.warning(f"Permission denied listing {kind}: {e.reason}")
                permission_notice = {
                    "permission_limited": True,
                    "resource": kind,
                    "namespace": namespace or "cluster-wide",
                    "denied_actions": ["list"],
                    "message": f"⚠️  Access denied: {e.reason}",
                    "recommendation": f"Request 'list' permission for {kind} resources",
                }
                return [], permission_notice

            logger.error(f"Error listing resources: {e}")
            return [], {"error": str(e)}

    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear_all()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    async def get_accessible_namespaces(self) -> List[str]:
        """Get list of namespaces user can access."""
        if not self.permission_checker:
            try:
                namespaces = self.core_v1.list_namespace()
                return [ns.metadata.name for ns in namespaces.items]
            except:
                return []

        accessible = await self.permission_checker.get_accessible_namespaces()
        return list(accessible)
