import logging
from typing import Any, Dict, List

from ...models import ResourceIdentifier
from ...relationships import RelationshipType, ResourceRelationship

logger = logging.getLogger(__name__)


class NativeResourceDiscoverer:

    def __init__(self, client):
        self.client = client

    async def discover(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        relationships = []
        kind = resource.get("kind")

        relationships.extend(await self._discover_owner_refs(resource))
        relationships.extend(await self._discover_owned_resources(resource))

        if kind == "Pod":
            relationships.extend(await self._discover_pod_relationships(resource))
        elif kind == "Service":
            relationships.extend(await self._discover_service_relationships(resource))
        elif kind == "Ingress":
            relationships.extend(await self._discover_ingress_relationships(resource))
        elif kind in ["Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"]:
            relationships.extend(await self._discover_workload_relationships(resource))
        elif kind == "PersistentVolumeClaim":
            relationships.extend(await self._discover_pvc_relationships(resource))

        return relationships

    async def _discover_owner_refs(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
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
                    details=f"Owned by {owner.get('kind')}",
                )
            )

        return relationships

    async def _discover_owned_resources(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        kind = resource.get("kind")

        search_kinds = {
            "Deployment": ["ReplicaSet"],
            "ReplicaSet": ["Pod"],
            "StatefulSet": ["Pod"],
            "DaemonSet": ["Pod"],
            "Job": ["Pod"],
            "CronJob": ["Job"],
        }.get(kind, [])

        for search_kind in search_kinds:
            try:
                resources, _ = await self.client.list_resources(
                    kind=search_kind, namespace=resource_id.namespace, use_cache=True
                )

                for res in resources:
                    res_metadata = res.get("metadata", {})
                    owner_refs = res_metadata.get("ownerReferences", [])

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
                                    details=f"Owns {search_kind}",
                                )
                            )
            except Exception as e:
                logger.warning(f"Error discovering {search_kind}: {e}")

        return relationships

    async def _discover_pod_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})

        for volume in spec.get("volumes", []):
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

            elif "persistentVolumeClaim" in volume:
                pvc_name = volume["persistentVolumeClaim"].get("claimName")
                if pvc_name:
                    pvc_id = ResourceIdentifier(
                        kind="PersistentVolumeClaim", name=pvc_name, namespace=resource_id.namespace
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=pvc_id,
                            relationship_type=RelationshipType.VOLUME,
                            details=f"Uses PVC {volume.get('name')}",
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

        node_name = spec.get("nodeName")
        if node_name:
            node_id = ResourceIdentifier(kind="Node", name=node_name, namespace=None)
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=node_id,
                    relationship_type=RelationshipType.REFERENCE,
                    details=f"Scheduled on node {node_name}",
                )
            )

        return relationships

    async def _discover_service_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})
        selector = spec.get("selector", {})

        if selector:
            label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

            try:
                pods, _ = await self.client.list_resources(
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
                            details="Service routes to Pod via selector",
                        )
                    )
            except Exception as e:
                logger.debug(f"Error discovering pods for service: {e}")

        return relationships

    async def _discover_ingress_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
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
                            details="Ingress routes to Service",
                        )
                    )

        return relationships

    async def _discover_workload_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})
        selector = spec.get("selector", {}).get("matchLabels", {})

        if selector:
            label_selector = ",".join(f"{k}={v}" for k, v in selector.items())

            try:
                pods, _ = await self.client.list_resources(
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

    async def _discover_pvc_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})

        storage_class = spec.get("storageClassName")
        if storage_class:
            sc_id = ResourceIdentifier(kind="StorageClass", name=storage_class, namespace=None)
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=sc_id,
                    relationship_type=RelationshipType.REFERENCE,
                    details=f"Uses StorageClass {storage_class}",
                )
            )

        volume_name = spec.get("volumeName")
        if volume_name:
            pv_id = ResourceIdentifier(kind="PersistentVolume", name=volume_name, namespace=None)
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=pv_id,
                    relationship_type=RelationshipType.REFERENCE,
                    details=f"Bound to PV {volume_name}",
                )
            )

        return relationships

    def _extract_resource_id(self, resource: Dict[str, Any]) -> ResourceIdentifier:
        metadata = resource.get("metadata", {})
        return ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )
