import logging
from typing import Any, Dict, List

from ...models import ResourceIdentifier
from ...relationships import RelationshipType, ResourceRelationship

logger = logging.getLogger(__name__)


class NetworkPolicyDiscoverer:

    def __init__(self, client):
        self.client = client

    async def discover(self, resource: Dict[str, Any]) -> List[ResourceRelationship]:
        relationships = []
        kind = resource.get("kind")

        if kind == "NetworkPolicy":
            relationships.extend(await self._discover_networkpolicy_relationships(resource))

        return relationships

    async def _discover_networkpolicy_relationships(
        self, resource: Dict[str, Any]
    ) -> List[ResourceRelationship]:
        relationships = []
        resource_id = self._extract_resource_id(resource)
        spec = resource.get("spec", {})

        pod_selector = spec.get("podSelector", {})
        match_labels = pod_selector.get("matchLabels", {})

        if match_labels:
            label_selector = ",".join(f"{k}={v}" for k, v in match_labels.items())
        elif not pod_selector or pod_selector == {}:
            label_selector = None
        else:
            label_selector = None

        try:
            if label_selector:
                pods, _ = await self.client.list_resources(
                    kind="Pod",
                    namespace=resource_id.namespace,
                    label_selector=label_selector,
                    use_cache=True,
                )
            else:
                pods, _ = await self.client.list_resources(
                    kind="Pod", namespace=resource_id.namespace, use_cache=True
                )

            for pod in pods[:50]:
                pod_id = self._extract_resource_id(pod)
                relationships.append(
                    ResourceRelationship(
                        source=resource_id,
                        target=pod_id,
                        relationship_type=RelationshipType.NETWORK,
                        details="NetworkPolicy applies to Pod",
                    )
                )
        except Exception as e:
            logger.debug(f"Error discovering pods for NetworkPolicy: {e}")

        for ingress_rule in spec.get("ingress", []):
            for from_rule in ingress_rule.get("from", []):
                if "namespaceSelector" in from_rule:
                    ns_selector = from_rule["namespaceSelector"]
                    match_labels = ns_selector.get("matchLabels", {})
                    if match_labels:
                        try:
                            namespaces, _ = await self.client.list_resources(
                                kind="Namespace", namespace=None, use_cache=True
                            )

                            for ns in namespaces:
                                ns_labels = ns.get("metadata", {}).get("labels", {})
                                if all(ns_labels.get(k) == v for k, v in match_labels.items()):
                                    ns_id = self._extract_resource_id(ns)
                                    relationships.append(
                                        ResourceRelationship(
                                            source=resource_id,
                                            target=ns_id,
                                            relationship_type=RelationshipType.NETWORK,
                                            details="Allows ingress from Namespace",
                                        )
                                    )
                        except Exception as e:
                            logger.debug(f"Error discovering namespaces for NetworkPolicy: {e}")

        for egress_rule in spec.get("egress", []):
            for to_rule in egress_rule.get("to", []):
                if "namespaceSelector" in to_rule:
                    ns_selector = to_rule["namespaceSelector"]
                    match_labels = ns_selector.get("matchLabels", {})
                    if match_labels:
                        try:
                            namespaces, _ = await self.client.list_resources(
                                kind="Namespace", namespace=None, use_cache=True
                            )

                            for ns in namespaces:
                                ns_labels = ns.get("metadata", {}).get("labels", {})
                                if all(ns_labels.get(k) == v for k, v in match_labels.items()):
                                    ns_id = self._extract_resource_id(ns)
                                    relationships.append(
                                        ResourceRelationship(
                                            source=resource_id,
                                            target=ns_id,
                                            relationship_type=RelationshipType.NETWORK,
                                            details="Allows egress to Namespace",
                                        )
                                    )
                        except Exception as e:
                            logger.debug(f"Error discovering namespaces for NetworkPolicy: {e}")

        return relationships

    def _extract_resource_id(self, resource: Dict[str, Any]) -> ResourceIdentifier:
        metadata = resource.get("metadata", {})
        return ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )
