import logging
from typing import Any, Dict, List

from ...models import ResourceIdentifier
from ...relationships import RelationshipType, ResourceRelationship

logger = logging.getLogger(__name__)


class RBACDiscoverer:

    def __init__(self, client):
        self.client = client

    async def discover(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        relationships = []
        kind = resource.get("kind")

        if kind == "Pod":
            relationships.extend(await self._discover_pod_rbac(resource))
        elif kind == "ServiceAccount":
            relationships.extend(await self._discover_serviceaccount_rbac(resource))
        elif kind in ["RoleBinding", "ClusterRoleBinding"]:
            relationships.extend(await self._discover_rolebinding_rbac(resource))

        return relationships

    async def _discover_pod_rbac(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})

        service_account = spec.get("serviceAccountName", "default")
        if service_account:
            sa_id = ResourceIdentifier(
                kind="ServiceAccount", name=service_account, namespace=resource_id.namespace
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=sa_id,
                    relationship_type=RelationshipType.REFERENCE,
                    details=f"Uses ServiceAccount {service_account}",
                )
            )

        for volume in spec.get("volumes", []):
            if "secret" in volume:
                secret_name = volume["secret"].get("secretName", "")
                if "token" in secret_name or "sa-token" in secret_name:
                    secret_id = ResourceIdentifier(
                        kind="Secret", name=secret_name, namespace=resource_id.namespace
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=secret_id,
                            relationship_type=RelationshipType.REFERENCE,
                            details="ServiceAccount token",
                        )
                    )

        return relationships

    async def _discover_serviceaccount_rbac(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        namespace = resource_id.namespace
        sa_name = resource_id.name

        try:
            rolebindings, _ = await self.client.list_resources(
                kind="RoleBinding", namespace=namespace, use_cache=True
            )

            for rb in rolebindings:
                subjects = rb.get("subjects", [])
                for subject in subjects:
                    if subject.get("kind") == "ServiceAccount" and subject.get("name") == sa_name:

                        rb_id = self._extract_resource_id(rb)
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=rb_id,
                                relationship_type=RelationshipType.REFERENCE,
                                details="Bound to RoleBinding",
                            )
                        )

                        role_ref = rb.get("roleRef", {})
                        role_name = role_ref.get("name")
                        role_kind = role_ref.get("kind", "Role")
                        if role_name:
                            role_id = ResourceIdentifier(
                                kind=role_kind,
                                name=role_name,
                                namespace=namespace if role_kind == "Role" else None,
                            )
                            relationships.append(
                                ResourceRelationship(
                                    source=rb_id,
                                    target=role_id,
                                    relationship_type=RelationshipType.REFERENCE,
                                    details=f"Grants {role_kind} permissions",
                                )
                            )
        except Exception as e:
            logger.debug(f"Error discovering RoleBindings: {e}")

        try:
            clusterrolebindings, _ = await self.client.list_resources(
                kind="ClusterRoleBinding", namespace=None, use_cache=True
            )

            for crb in clusterrolebindings:
                subjects = crb.get("subjects", [])
                for subject in subjects:
                    if (
                        subject.get("kind") == "ServiceAccount"
                        and subject.get("name") == sa_name
                        and subject.get("namespace") == namespace
                    ):

                        crb_id = self._extract_resource_id(crb)
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=crb_id,
                                relationship_type=RelationshipType.REFERENCE,
                                details="Bound to ClusterRoleBinding",
                            )
                        )

                        role_ref = crb.get("roleRef", {})
                        role_name = role_ref.get("name")
                        if role_name:
                            role_id = ResourceIdentifier(
                                kind="ClusterRole", name=role_name, namespace=None
                            )
                            relationships.append(
                                ResourceRelationship(
                                    source=crb_id,
                                    target=role_id,
                                    relationship_type=RelationshipType.REFERENCE,
                                    details="Grants ClusterRole permissions",
                                )
                            )
        except Exception as e:
            logger.debug(f"Error discovering ClusterRoleBindings: {e}")

        return relationships

    async def _discover_rolebinding_rbac(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)

        subjects = resource.get("subjects", [])
        for subject in subjects:
            subject_kind = subject.get("kind")
            subject_name = subject.get("name")
            subject_namespace = subject.get("namespace", resource_id.namespace)

            if subject_kind in ["ServiceAccount", "User", "Group"]:
                subject_id = ResourceIdentifier(
                    kind=subject_kind,
                    name=subject_name,
                    namespace=subject_namespace if subject_kind == "ServiceAccount" else None,
                )
                relationships.append(
                    ResourceRelationship(
                        source=resource_id,
                        target=subject_id,
                        relationship_type=RelationshipType.REFERENCE,
                        details=f"Binds to {subject_kind}",
                    )
                )

        role_ref = resource.get("roleRef", {})
        role_kind = role_ref.get("kind")
        role_name = role_ref.get("name")

        if role_name:
            role_namespace = resource_id.namespace if role_kind == "Role" else None
            role_id = ResourceIdentifier(kind=role_kind, name=role_name, namespace=role_namespace)
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=role_id,
                    relationship_type=RelationshipType.REFERENCE,
                    details=f"References {role_kind}",
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
