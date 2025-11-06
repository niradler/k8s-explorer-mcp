"""Resource relationship discovery engine."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .models import RelationshipType, ResourceIdentifier

logger = logging.getLogger(__name__)


@dataclass
class ResourceRelationship:
    """Represents a relationship between two resources."""

    source: ResourceIdentifier
    target: ResourceIdentifier
    relationship_type: RelationshipType
    details: Optional[str] = None


class RelationshipDiscovery:
    """Discovers relationships between Kubernetes resources."""

    def __init__(self, client):
        """Initialize with K8s client."""
        self.client = client

    async def discover_relationships(
        self, resource: Dict[str, Any], max_depth: int = 3
    ) -> List[ResourceRelationship]:
        """
        Discover all relationships for a resource.

        Discovers:
        - Owner references (parent resources)
        - Owned resources (children)
        - Label selectors (services -> pods)
        - Volume mounts (configmaps, secrets)
        - Service endpoints
        - Ingress backends
        - Helm chart information
        """
        relationships = []
        resource_id = self._extract_resource_id(resource)

        relationships.extend(await self._discover_owner_refs(resource))
        relationships.extend(await self._discover_owned_resources(resource_id))
        relationships.extend(await self._discover_label_selectors(resource))
        relationships.extend(await self._discover_volumes(resource))
        relationships.extend(await self._discover_service_refs(resource))
        relationships.extend(self._discover_helm_info(resource))

        return relationships

    async def _discover_owner_refs(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        """Discover parent resources via ownerReferences."""
        relationships = []
        metadata = resource.get("metadata", {})
        owner_refs = metadata.get("ownerReferences", [])

        resource_id = self._extract_resource_id(resource)

        for owner in owner_refs:
            parent_id = ResourceIdentifier(
                kind=owner.get("kind"),
                name=owner.get("name"),
                namespace=metadata.get("namespace"),
                api_version=owner.get("apiVersion"),
            )

            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=parent_id,
                    relationship_type=RelationshipType.OWNER,
                    details=f"Controller: {owner.get('controller', False)}",
                )
            )

        return relationships

    async def _discover_owned_resources(
        self, resource_id: ResourceIdentifier
    ) -> List[ResourceRelationship]:
        """Discover child resources that reference this resource."""
        relationships = []

        search_kinds = ["Pod", "ReplicaSet", "Job", "Service"]

        for kind in search_kinds:
            try:
                resources = await self.client.list_resources(
                    kind=kind, namespace=resource_id.namespace, use_cache=True
                )

                for res in resources:
                    metadata = res.get("metadata", {})
                    owner_refs = metadata.get("ownerReferences", [])

                    for owner in owner_refs:
                        if (
                            owner.get("name") == resource_id.name
                            and owner.get("kind") == resource_id.kind
                        ):

                            child_id = self._extract_resource_id(res)
                            relationships.append(
                                ResourceRelationship(
                                    source=resource_id,
                                    target=child_id,
                                    relationship_type=RelationshipType.OWNED,
                                    details=f"Owns {kind}",
                                )
                            )
            except Exception as e:
                logger.debug(f"Error discovering {kind}: {e}")

        return relationships

    async def _discover_label_selectors(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        """Discover resources via label selectors."""
        relationships = []
        kind = resource.get("kind")
        resource_id = self._extract_resource_id(resource)

        if kind == "Service":
            selector = resource.get("spec", {}).get("selector", {})
            if selector:
                label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

                try:
                    pods = await self.client.list_resources(
                        kind="Pod",
                        namespace=resource_id.namespace,
                        label_selector=label_selector,
                        use_cache=True,
                    )

                    for pod in pods:
                        pod_id = self._extract_resource_id(pod)
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=pod_id,
                                relationship_type=RelationshipType.SELECTOR,
                                details=f"Service selector: {label_selector}",
                            )
                        )
                except Exception as e:
                    logger.debug(f"Error discovering pods for service: {e}")

        elif kind in ["Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"]:
            selector = resource.get("spec", {}).get("selector", {}).get("matchLabels", {})
            if selector:
                label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

                try:
                    pods = await self.client.list_resources(
                        kind="Pod",
                        namespace=resource_id.namespace,
                        label_selector=label_selector,
                        use_cache=True,
                    )

                    for pod in pods[:20]:
                        pod_id = self._extract_resource_id(pod)
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=pod_id,
                                relationship_type=RelationshipType.SELECTOR,
                                details="Manages pod via selector",
                            )
                        )
                except Exception as e:
                    logger.debug(f"Error discovering pods: {e}")

        return relationships

    async def _discover_volumes(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        """Discover ConfigMaps and Secrets used in volumes."""
        relationships = []
        kind = resource.get("kind")

        if kind != "Pod":
            return relationships

        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})
        volumes = spec.get("volumes", [])

        for volume in volumes:
            if "configMap" in volume:
                cm_name = volume["configMap"].get("name")
                if cm_name:
                    cm_id = ResourceIdentifier(
                        kind="ConfigMap", name=cm_name, namespace=resource_id.namespace
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=cm_id,
                            relationship_type=RelationshipType.VOLUME,
                            details=f"Mounts ConfigMap as {volume.get('name')}",
                        )
                    )

            elif "secret" in volume:
                secret_name = volume["secret"].get("secretName")
                if secret_name:
                    secret_id = ResourceIdentifier(
                        kind="Secret", name=secret_name, namespace=resource_id.namespace
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=secret_id,
                            relationship_type=RelationshipType.VOLUME,
                            details=f"Mounts Secret as {volume.get('name')}",
                        )
                    )

        for container in spec.get("containers", []):
            for env in container.get("env", []):
                value_from = env.get("valueFrom", {})

                if "configMapKeyRef" in value_from:
                    cm_name = value_from["configMapKeyRef"].get("name")
                    if cm_name:
                        cm_id = ResourceIdentifier(
                            kind="ConfigMap", name=cm_name, namespace=resource_id.namespace
                        )
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=cm_id,
                                relationship_type=RelationshipType.REFERENCE,
                                details="Env var from ConfigMap",
                            )
                        )

                elif "secretKeyRef" in value_from:
                    secret_name = value_from["secretKeyRef"].get("name")
                    if secret_name:
                        secret_id = ResourceIdentifier(
                            kind="Secret", name=secret_name, namespace=resource_id.namespace
                        )
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=secret_id,
                                relationship_type=RelationshipType.REFERENCE,
                                details="Env var from Secret",
                            )
                        )

        return relationships

    async def _discover_service_refs(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        """Discover service references from Ingress."""
        relationships = []
        kind = resource.get("kind")

        if kind != "Ingress":
            return relationships

        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})

        for rule in spec.get("rules", []):
            http = rule.get("http", {})
            for path in http.get("paths", []):
                backend = path.get("backend", {})
                service = backend.get("service", {})
                service_name = service.get("name")

                if service_name:
                    service_id = ResourceIdentifier(
                        kind="Service", name=service_name, namespace=resource_id.namespace
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=service_id,
                            relationship_type=RelationshipType.SERVICE,
                            details="Routes to service",
                        )
                    )

        return relationships

    def _extract_resource_id(self, resource: Dict[str, Any]) -> ResourceIdentifier:
        """Extract ResourceIdentifier from resource dict."""
        metadata = resource.get("metadata", {})
        return ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

    def _discover_helm_info(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        """
        Discover Helm chart information from resource metadata.

        Detects:
        - Helm release name
        - Chart name and version
        - Helm-managed resources
        """
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})

        resource_id = self._extract_resource_id(resource)

        helm_release = (
            labels.get("helm.sh/chart")
            or labels.get("app.kubernetes.io/managed-by") == "Helm"
            and labels.get("app.kubernetes.io/instance")
        )

        if helm_release or "helm.sh" in str(annotations):
            helm_details = {
                "managed_by": "Helm",
                "release": labels.get("app.kubernetes.io/instance") or labels.get("release"),
                "chart": labels.get("helm.sh/chart") or labels.get("chart"),
                "version": labels.get("app.kubernetes.io/version"),
                "heritage": labels.get("heritage"),
            }

            helm_details = {k: v for k, v in helm_details.items() if v}

            if helm_details:
                details_str = ", ".join(f"{k}: {v}" for k, v in helm_details.items())

                relationships.append(
                    ResourceRelationship(
                        source=resource_id,
                        target=ResourceIdentifier(
                            kind="HelmRelease",
                            name=helm_details.get("release", "unknown"),
                            namespace=resource_id.namespace,
                            api_version="helm.sh/v1",
                        ),
                        relationship_type=RelationshipType.CRD,
                        details=f"Helm: {details_str}",
                    )
                )

        return relationships

    def get_resource_summary_for_llm(
        self, resource: Dict[str, Any], relationships: List[ResourceRelationship]
    ) -> Dict[str, Any]:
        """
        Create comprehensive resource summary optimized for LLM consumption.

        Includes:
        - Resource identity
        - Helm/operator management info
        - All discovered relationships grouped by type
        - Dependency summary
        """
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})

        summary = {
            "identity": {
                "kind": resource.get("kind"),
                "name": metadata.get("name"),
                "namespace": metadata.get("namespace"),
                "created": metadata.get("creationTimestamp"),
            },
            "management": self._detect_management_info(labels, annotations),
            "relationships": self._group_relationships(relationships),
            "dependency_summary": self._create_dependency_summary(relationships),
        }

        return summary

    def _detect_management_info(
        self, labels: Dict[str, str], annotations: Dict[str, str]
    ) -> Dict[str, str]:
        """Detect how resource is managed (Helm, Operator, manual)."""
        info = {}

        managed_by = labels.get("app.kubernetes.io/managed-by")
        if managed_by:
            info["managed_by"] = managed_by

        if managed_by == "Helm" or "helm.sh" in str(labels) or "helm.sh" in str(annotations):
            info["type"] = "Helm"
            info["release"] = labels.get("app.kubernetes.io/instance") or labels.get("release")
            info["chart"] = labels.get("helm.sh/chart") or labels.get("chart")

        if "argocd.argoproj.io" in str(annotations):
            info["type"] = "ArgoCD"
            info["application"] = annotations.get("argocd.argoproj.io/instance")

        if "workflows.argoproj.io" in str(labels):
            info["type"] = "Argo Workflows"
            info["workflow"] = labels.get("workflows.argoproj.io/workflow")

        if "dag_id" in labels:
            info["type"] = "Airflow"
            info["dag"] = labels.get("dag_id")
            info["task"] = labels.get("task_id")

        if not info:
            info["type"] = "Manual"

        return info

    def _group_relationships(
        self, relationships: List[ResourceRelationship]
    ) -> Dict[str, List[Dict]]:
        """Group relationships by type for clear presentation."""
        grouped = {
            "created_by": [],
            "creates": [],
            "uses_config": [],
            "routes_to": [],
            "managed_by_operator": [],
        }

        for rel in relationships:
            rel_data = {"kind": rel.target.kind, "name": rel.target.name, "details": rel.details}

            if rel.relationship_type.value == "owner":
                grouped["created_by"].append(rel_data)
            elif rel.relationship_type.value == "owned":
                grouped["creates"].append(rel_data)
            elif rel.relationship_type.value in ["volume", "reference"]:
                grouped["uses_config"].append(rel_data)
            elif rel.relationship_type.value == "service":
                grouped["routes_to"].append(rel_data)
            elif rel.relationship_type.value == "crd":
                grouped["managed_by_operator"].append(rel_data)

        return {k: v for k, v in grouped.items() if v}

    def _create_dependency_summary(self, relationships: List[ResourceRelationship]) -> str:
        """Create human-readable dependency summary."""
        counts = {
            "parents": len([r for r in relationships if r.relationship_type.value == "owner"]),
            "children": len([r for r in relationships if r.relationship_type.value == "owned"]),
            "configs": len(
                [r for r in relationships if r.relationship_type.value in ["volume", "reference"]]
            ),
            "operators": len([r for r in relationships if r.relationship_type.value == "crd"]),
        }

        parts = []
        if counts["parents"]:
            parts.append(f"created by {counts['parents']} parent(s)")
        if counts["children"]:
            parts.append(f"manages {counts['children']} child resource(s)")
        if counts["configs"]:
            parts.append(f"uses {counts['configs']} config resource(s)")
        if counts["operators"]:
            parts.append(f"managed by {counts['operators']} operator(s)")

        if not parts:
            return "Standalone resource with no discovered dependencies"

        return "Resource " + ", ".join(parts)

    async def build_resource_tree(
        self, resource_id: ResourceIdentifier, max_depth: int = 3, visited: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Build complete resource tree starting from a resource."""
        if visited is None:
            visited = set()

        if max_depth <= 0:
            return {}

        key = (resource_id.kind, resource_id.name, resource_id.namespace)
        if key in visited:
            return {}

        visited.add(key)

        resource = await self.client.get_resource(resource_id, use_cache=True)
        if not resource:
            return {}

        relationships = await self.discover_relationships(resource, max_depth)

        tree = {"resource": self._summarize_resource(resource), "relationships": []}

        for rel in relationships:
            child_tree = await self.build_resource_tree(
                rel.target, max_depth=max_depth - 1, visited=visited
            )

            tree["relationships"].append(
                {
                    "type": rel.relationship_type.value,
                    "details": rel.details,
                    "target": child_tree if child_tree else self._summarize_resource_id(rel.target),
                }
            )

        return tree

    def _summarize_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal resource summary."""
        metadata = resource.get("metadata", {})
        return {
            "kind": resource.get("kind"),
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "labels": metadata.get("labels", {}),
            "created": metadata.get("creationTimestamp"),
        }

    def _summarize_resource_id(self, resource_id: ResourceIdentifier) -> Dict[str, str]:
        """Summarize resource identifier."""
        return {
            "kind": resource_id.kind,
            "name": resource_id.name,
            "namespace": resource_id.namespace or "cluster",
        }
