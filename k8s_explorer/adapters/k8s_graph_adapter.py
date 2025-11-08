from typing import Any, Dict, List, Optional

from k8s_graph.protocols import K8sClientProtocol
from k8s_graph.models import ResourceIdentifier as K8sGraphResourceIdentifier

from ..client import K8sClient
from ..models import ResourceIdentifier


class K8sGraphClientAdapter(K8sClientProtocol):

    def __init__(self, k8s_client: K8sClient):
        self.k8s_client = k8s_client

    async def get_resource(
        self, resource_id: K8sGraphResourceIdentifier
    ) -> Optional[Dict[str, Any]]:
        our_resource_id = ResourceIdentifier(
            kind=resource_id.kind,
            name=resource_id.name,
            namespace=resource_id.namespace,
            api_version=getattr(resource_id, "api_version", None),
        )

        try:
            resource = await self.k8s_client.get_resource(our_resource_id)
            return resource
        except Exception as e:
            return None

    async def list_resources(
        self,
        kind: str,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            resources = await self.k8s_client.list_resources(
                kind=kind, namespace=namespace, label_selector=label_selector
            )
            return resources if resources else []
        except Exception as e:
            return []
