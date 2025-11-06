"""Permission checking and RBAC awareness for Kubernetes."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Set

from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


@dataclass
class PermissionCheck:
    """Result of a permission check."""

    allowed: bool
    verb: str
    resource: str
    namespace: Optional[str]
    reason: Optional[str] = None


class PermissionChecker:
    """Checks and caches Kubernetes RBAC permissions."""

    def __init__(self, api_client: client.ApiClient):
        """Initialize permission checker."""
        self.auth_v1 = client.AuthorizationV1Api(api_client)
        self._permission_cache: Dict[str, PermissionCheck] = {}
        self._accessible_namespaces: Optional[Set[str]] = None

    async def can_access(
        self, verb: str, resource: str, namespace: Optional[str] = None
    ) -> PermissionCheck:
        """
        Check if current user can perform an action.

        Args:
            verb: Action (get, list, watch, create, delete, etc.)
            resource: Resource type (pods, services, deployments, etc.)
            namespace: Optional namespace (for namespaced resources)

        Returns:
            PermissionCheck with result and details
        """
        cache_key = f"{verb}:{resource}:{namespace or 'cluster'}"

        if cache_key in self._permission_cache:
            return self._permission_cache[cache_key]

        try:
            body = client.V1SelfSubjectAccessReview(
                spec=client.V1SelfSubjectAccessReviewSpec(
                    resource_attributes=client.V1ResourceAttributes(
                        verb=verb, resource=resource, namespace=namespace
                    )
                )
            )

            response = self.auth_v1.create_self_subject_access_review(body)

            result = PermissionCheck(
                allowed=response.status.allowed,
                verb=verb,
                resource=resource,
                namespace=namespace,
                reason=response.status.reason if not response.status.allowed else None,
            )

            self._permission_cache[cache_key] = result
            return result

        except ApiException as e:
            logger.warning(f"Permission check failed: {e}")
            return PermissionCheck(
                allowed=False,
                verb=verb,
                resource=resource,
                namespace=namespace,
                reason=f"Permission check failed: {e.reason}",
            )

    async def get_accessible_namespaces(self) -> Set[str]:
        """
        Get list of namespaces the user can list pods in.

        Returns:
            Set of accessible namespace names
        """
        if self._accessible_namespaces is not None:
            return self._accessible_namespaces

        accessible = set()

        try:
            v1 = client.CoreV1Api()

            can_list_all = await self.can_access("list", "namespaces")
            if not can_list_all.allowed:
                logger.warning("Cannot list cluster namespaces - limited namespace discovery")
                return accessible

            namespaces = v1.list_namespace()

            for ns in namespaces.items:
                ns_name = ns.metadata.name

                can_list_pods = await self.can_access("list", "pods", ns_name)
                if can_list_pods.allowed:
                    accessible.add(ns_name)

            self._accessible_namespaces = accessible
            logger.info(f"User has access to {len(accessible)} namespaces")

        except Exception as e:
            logger.error(f"Error discovering accessible namespaces: {e}")

        return accessible

    async def check_resource_access(
        self, kind: str, namespace: Optional[str] = None
    ) -> Dict[str, PermissionCheck]:
        """
        Check all relevant permissions for a resource type.

        Args:
            kind: Resource kind (Pod, Deployment, etc.)
            namespace: Optional namespace

        Returns:
            Dict of verb -> PermissionCheck
        """
        resource_map = {
            "Pod": "pods",
            "Service": "services",
            "Deployment": "deployments",
            "ReplicaSet": "replicasets",
            "StatefulSet": "statefulsets",
            "DaemonSet": "daemonsets",
            "Job": "jobs",
            "CronJob": "cronjobs",
            "ConfigMap": "configmaps",
            "Secret": "secrets",
            "Ingress": "ingresses",
            "PersistentVolumeClaim": "persistentvolumeclaims",
        }

        resource = resource_map.get(kind, kind.lower() + "s")
        verbs = ["get", "list", "watch"]

        checks = {}
        for verb in verbs:
            checks[verb] = await self.can_access(verb, resource, namespace)

        return checks

    def get_permission_summary(self, checks: Dict[str, PermissionCheck]) -> str:
        """
        Generate human-readable permission summary.

        Args:
            checks: Dict of verb -> PermissionCheck

        Returns:
            Summary string
        """
        allowed = [verb for verb, check in checks.items() if check.allowed]
        denied = [verb for verb, check in checks.items() if not check.allowed]

        if not denied:
            return "Full access granted"
        elif not allowed:
            return "No access - insufficient permissions"
        else:
            return f"Limited access: can {', '.join(allowed)} but cannot {', '.join(denied)}"

    def create_permission_notice(
        self, resource_kind: str, namespace: Optional[str], checks: Dict[str, PermissionCheck]
    ) -> Dict[str, any]:
        """
        Create permission notice for API responses.

        Args:
            resource_kind: Kind of resource
            namespace: Namespace being accessed
            checks: Permission check results

        Returns:
            Permission notice dict
        """
        denied_verbs = [verb for verb, check in checks.items() if not check.allowed]

        if not denied_verbs:
            return None

        notice = {
            "permission_limited": True,
            "resource": resource_kind,
            "namespace": namespace or "cluster-wide",
            "denied_actions": denied_verbs,
            "summary": self.get_permission_summary(checks),
        }

        if "list" in denied_verbs:
            notice["message"] = (
                f"⚠️  Cannot list {resource_kind} resources. "
                f"Results may be incomplete or unavailable."
            )
            notice["recommendation"] = (
                f"Request 'list' permission for {checks['list'].resource} "
                f"in namespace '{namespace or 'cluster-wide'}'"
            )
        elif "get" in denied_verbs:
            notice["message"] = (
                f"⚠️  Cannot get individual {resource_kind} resources. " f"Can only list resources."
            )

        return notice

    def clear_cache(self):
        """Clear permission cache."""
        self._permission_cache.clear()
        self._accessible_namespaces = None
        logger.info("Permission cache cleared")
