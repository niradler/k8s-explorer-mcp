import logging
from typing import Any, Dict, List

from ...operators.crd_handlers import CRDOperatorRegistry
from ...relationships import ResourceRelationship

logger = logging.getLogger(__name__)


class CustomResourceDiscoverer:

    def __init__(self, client):
        self.client = client
        self.crd_registry = CRDOperatorRegistry()

    async def discover(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        relationships = []

        api_version = resource.get("apiVersion", "")
        kind = resource.get("kind", "")

        for handler in self.crd_registry.handlers:
            if handler.can_handle(resource):
                try:
                    handler_relationships = await handler.discover_relationships(
                        resource, self.client
                    )
                    relationships.extend(handler_relationships)
                    logger.debug(
                        f"CRD handler {handler.__class__.__name__} found "
                        f"{len(handler_relationships)} relationships for {kind}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error in CRD handler {handler.__class__.__name__} " f"for {kind}: {e}"
                    )

        return relationships
