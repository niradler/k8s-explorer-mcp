import logging
from typing import Any, Dict, List

from k8s_graph.discoverers import BaseDiscoverer
from k8s_graph.models import ResourceIdentifier as K8sGraphResourceIdentifier

from ..models import RelationshipType, ResourceIdentifier
from ..operators.crd_handlers import CRDOperatorRegistry
from ..relationships import ResourceRelationship

logger = logging.getLogger(__name__)


class CRDDiscovererAdapter(BaseDiscoverer):

    def __init__(self, client, crd_registry: CRDOperatorRegistry):
        super().__init__(client)
        self.crd_registry = crd_registry

    def supports(self, resource: Dict[str, Any]) -> bool:
        kind = resource.get("kind", "")
        api_version = resource.get("apiVersion", "")

        if not api_version or not kind:
            return False

        if "/" not in api_version:
            return False

        for handler in self.crd_registry.handlers:
            if handler.can_handle(resource):
                return True

        return False

    async def discover(self, resource: Dict[str, Any]) -> List[tuple]:
        our_relationships = await self.crd_registry.discover_crd_relationships(
            resource, self.client.k8s_client if hasattr(self.client, "k8s_client") else self.client
        )

        k8s_graph_relationships = []
        for rel in our_relationships:
            source_id = self._convert_resource_id(rel.source)
            target_id = self._convert_resource_id(rel.target)
            relationship_type = self._convert_relationship_type(rel.relationship_type)

            k8s_graph_relationships.append((source_id, target_id, relationship_type, rel.details or ""))

        return k8s_graph_relationships

    def _convert_resource_id(self, our_id: ResourceIdentifier) -> K8sGraphResourceIdentifier:
        return K8sGraphResourceIdentifier(
            kind=our_id.kind, name=our_id.name, namespace=our_id.namespace
        )

    def _convert_relationship_type(self, our_type: RelationshipType) -> str:
        type_map = {
            "owner": "owner",
            "owned": "owned",
            "label_selector": "label_selector",
            "volume": "volume",
            "env_var": "env_var",
            "env_from": "env_from",
            "service_account": "service_account",
            "role_binding": "role_binding",
            "network_policy": "network_policy",
            "ingress_backend": "ingress_backend",
            "pvc": "pvc",
            "crd": "crd",
            "namespace": "namespace",
        }
        if isinstance(our_type, RelationshipType):
            return type_map.get(our_type.value, "crd")
        return type_map.get(str(our_type).lower(), "crd")
