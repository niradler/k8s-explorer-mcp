import logging
from typing import Any, Dict, List

from ...relationships import ResourceRelationship
from ..models import DiscoveryOptions
from .custom import CustomResourceDiscoverer
from .native import NativeResourceDiscoverer
from .network import NetworkPolicyDiscoverer
from .rbac import RBACDiscoverer

logger = logging.getLogger(__name__)


class UnifiedDiscoverer:

    def __init__(self, client):
        self.client = client
        self.native = NativeResourceDiscoverer(client)
        self.rbac = RBACDiscoverer(client)
        self.network = NetworkPolicyDiscoverer(client)
        self.custom = CustomResourceDiscoverer(client)

    async def discover_all_relationships(
        self, resource: Dict[str, Any], options: DiscoveryOptions
    ) -> List[ResourceRelationship]:
        relationships = []

        try:
            native_rels = await self.native.discover(resource)
            relationships.extend(native_rels)
            logger.debug(f"Native discoverer found {len(native_rels)} relationships")
        except Exception as e:
            logger.warning(f"Error in native discovery: {e}")

        if options.include_rbac:
            try:
                rbac_rels = await self.rbac.discover(resource)
                relationships.extend(rbac_rels)
                logger.debug(f"RBAC discoverer found {len(rbac_rels)} relationships")
            except Exception as e:
                logger.warning(f"Error in RBAC discovery: {e}")

        if options.include_network:
            try:
                network_rels = await self.network.discover(resource)
                relationships.extend(network_rels)
                logger.debug(f"Network discoverer found {len(network_rels)} relationships")
            except Exception as e:
                logger.warning(f"Error in network discovery: {e}")

        if options.include_crds:
            try:
                custom_rels = await self.custom.discover(resource)
                relationships.extend(custom_rels)
                logger.debug(f"Custom discoverer found {len(custom_rels)} relationships")
            except Exception as e:
                logger.warning(f"Error in custom discovery: {e}")

        return relationships


__all__ = [
    "UnifiedDiscoverer",
    "NativeResourceDiscoverer",
    "RBACDiscoverer",
    "NetworkPolicyDiscoverer",
    "CustomResourceDiscoverer",
]
