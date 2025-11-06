import asyncio
import logging
from typing import List, Optional, Set

import networkx as nx
from kubernetes.client.exceptions import ApiException

from ..models import ResourceIdentifier
from .cache import GraphCache
from .discoverers import UnifiedDiscoverer
from .models import BuildOptions, DiscoveryOptions
from .node_identity import NodeIdentity

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, client, cache: GraphCache):
        self.client = client
        self.cache = cache
        self.unified_discoverer = UnifiedDiscoverer(client)
        self.node_identity = NodeIdentity()
        self.permission_errors = []
        self.discovery_stats = {
            "total_relationships_discovered": 0,
            "nodes_processed": 0,
            "edges_added": 0,
        }

    async def build_from_resource(
        self, resource_id: ResourceIdentifier, depth: int, options: BuildOptions
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        visited: Set[str] = set()
        self.permission_errors = []
        self.discovery_stats = {
            "total_relationships_discovered": 0,
            "nodes_processed": 0,
            "edges_added": 0,
        }

        # Get cached graph to check for existing nodes
        namespace = resource_id.namespace or "cluster"
        cluster_id = options.cluster_id or "default"
        cached_data = self.cache.get_namespace_graph(namespace, cluster_id)
        cached_graph = cached_data[0] if cached_data else None

        discovery_options = DiscoveryOptions(
            include_rbac=options.include_rbac,
            include_network=options.include_network,
            include_crds=options.include_crds,
        )

        await self._expand_from_node(
            graph, resource_id, depth, visited, discovery_options, options, cached_graph
        )

        logger.info(
            f"Built graph from {resource_id.kind}/{resource_id.name}: "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        return graph

    async def build_namespace_graph(
        self, namespace: str, depth: int, options: BuildOptions
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        self.permission_errors = []

        # Get cached graph to check for existing nodes
        cluster_id = options.cluster_id or "default"
        cached_data = self.cache.get_namespace_graph(namespace, cluster_id)
        cached_graph = cached_data[0] if cached_data else None

        logger.info(f"Building full namespace graph for {namespace}")

        resource_types = [
            "Deployment",
            "StatefulSet",
            "DaemonSet",
            "ReplicaSet",
            "Pod",
            "Service",
            "Ingress",
            "ConfigMap",
            "Secret",
            "PersistentVolumeClaim",
            "ServiceAccount",
            "Role",
            "RoleBinding",
            "NetworkPolicy",
        ]

        if options.include_crds:
            resource_types.extend(
                [
                    "Workflow",
                    "CronWorkflow",
                    "Application",
                ]
            )

        all_resources = []
        for kind in resource_types:
            try:
                resources, _ = await self.client.list_resources(
                    kind=kind, namespace=namespace, use_cache=True
                )
                all_resources.extend(resources)
                logger.debug(f"Found {len(resources)} {kind} resources")
            except ApiException as e:
                if e.status == 403:
                    self.permission_errors.append(f"Access denied to {kind} resources")
                    logger.warning(f"Permission denied for {kind}: {e}")
                elif e.status == 404:
                    logger.debug(f"{kind} resource type not found")
                else:
                    logger.warning(f"Error listing {kind}: {e}")
            except Exception as e:
                logger.warning(f"Error listing {kind}: {e}")

        logger.info(f"Found {len(all_resources)} total resources in namespace {namespace}")

        visited: Set[str] = set()
        discovery_options = DiscoveryOptions(
            include_rbac=options.include_rbac,
            include_network=options.include_network,
            include_crds=options.include_crds,
        )

        for resource in all_resources[: options.max_nodes]:
            metadata = resource.get("metadata", {})
            resource_id = ResourceIdentifier(
                kind=resource.get("kind"),
                name=metadata.get("name"),
                namespace=metadata.get("namespace"),
                api_version=resource.get("apiVersion"),
            )

            await self._expand_from_node(
                graph, resource_id, min(depth, 2), visited, discovery_options, options, cached_graph
            )

        logger.info(
            f"Built namespace graph for {namespace}: "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

        return graph

    async def _expand_from_node(
        self,
        graph: nx.DiGraph,
        resource_id: ResourceIdentifier,
        depth: int,
        visited: Set[str],
        discovery_options: DiscoveryOptions,
        build_options: BuildOptions,
        cached_graph: Optional[nx.DiGraph] = None,
    ):
        if depth <= 0:
            return

        if graph.number_of_nodes() >= build_options.max_nodes:
            logger.warning(f"Reached max nodes limit ({build_options.max_nodes})")
            return

        try:
            resource = await self.client.get_resource(resource_id, use_cache=True)
            if not resource:
                logger.debug(f"Resource not found: {resource_id.kind}/{resource_id.name}")
                return
        except ApiException as e:
            if e.status == 403:
                self.permission_errors.append(
                    f"Access denied to {resource_id.kind}/{resource_id.name}"
                )
                logger.warning(f"Permission denied for {resource_id.kind}/{resource_id.name}")
            else:
                logger.warning(f"Error fetching {resource_id.kind}/{resource_id.name}: {e}")
            return
        except Exception as e:
            logger.warning(f"Error fetching {resource_id.kind}/{resource_id.name}: {e}")
            return

        node_id = self.node_identity.get_node_id(resource)

        if node_id in visited:
            return

        visited.add(node_id)

        node_attrs = self.node_identity.extract_node_attributes(resource)
        if build_options.cluster_id:
            node_attrs["cluster_id"] = build_options.cluster_id

        # Check for ANY duplicate node with same kind/namespace/name that we need to migrate
        metadata = resource.get("metadata", {})
        kind = resource.get("kind")
        name = metadata.get("name")
        namespace = metadata.get("namespace") or "cluster"

        # Find all nodes that match this resource
        duplicate_nodes = []
        for existing_node_id in list(graph.nodes()):
            if existing_node_id != node_id:
                attrs = graph.nodes[existing_node_id]
                existing_ns = attrs.get("namespace") or "cluster"
                if (
                    attrs.get("kind") == kind
                    and attrs.get("name") == name
                    and existing_ns == namespace
                ):
                    duplicate_nodes.append(existing_node_id)

        # Migrate all duplicates
        for old_node_id in duplicate_nodes:
            logger.info(f"Merging duplicate node {old_node_id} into {node_id}")

            # Transfer all edges
            for pred in list(graph.predecessors(old_node_id)):
                edge_data = graph.get_edge_data(pred, old_node_id)
                if not graph.has_edge(pred, node_id):
                    graph.add_edge(pred, node_id, **edge_data)

            for succ in list(graph.successors(old_node_id)):
                edge_data = graph.get_edge_data(old_node_id, succ)
                if not graph.has_edge(node_id, succ):
                    graph.add_edge(node_id, succ, **edge_data)

            # Remove old duplicate
            graph.remove_node(old_node_id)

        # Update node with full attributes (might be placeholder or new)
        if graph.has_node(node_id):
            # Update existing node with full attributes
            graph.nodes[node_id].update(node_attrs)
        else:
            # Add new node
            graph.add_node(node_id, **node_attrs)

        self.discovery_stats["nodes_processed"] += 1

        try:
            relationships = await self.unified_discoverer.discover_all_relationships(
                resource, discovery_options
            )
            self.discovery_stats["total_relationships_discovered"] += len(relationships)
            logger.info(f"Discovered {len(relationships)} relationships for {node_id}")
        except Exception as e:
            logger.warning(f"Error discovering relationships for {node_id}: {e}")
            relationships = []

        neighbor_tasks = []
        for rel in relationships:
            target_node_id = self._get_node_id_from_identifier(rel.target)

            # Check if a canonical node for this resource already exists
            # First check cached graph, then current subgraph
            canonical_node_id = None
            target_ns = rel.target.namespace or "cluster"

            # Check cached graph first (highest priority)
            if cached_graph:
                for existing_node_id in cached_graph.nodes():
                    attrs = cached_graph.nodes[existing_node_id]
                    existing_ns = attrs.get("namespace") or "cluster"
                    if (
                        attrs.get("kind") == rel.target.kind
                        and attrs.get("name") == rel.target.name
                        and existing_ns == target_ns
                    ):
                        canonical_node_id = existing_node_id
                        logger.debug(
                            f"Found canonical node {existing_node_id} in cache for {rel.target.kind}/{rel.target.name}"
                        )
                        break

            # If not in cache, check current subgraph
            if not canonical_node_id:
                for existing_node_id in graph.nodes():
                    attrs = graph.nodes[existing_node_id]
                    existing_ns = attrs.get("namespace") or "cluster"
                    if (
                        attrs.get("kind") == rel.target.kind
                        and attrs.get("name") == rel.target.name
                        and existing_ns == target_ns
                    ):
                        canonical_node_id = existing_node_id
                        logger.debug(
                            f"Found canonical node {existing_node_id} in subgraph for {rel.target.kind}/{rel.target.name}"
                        )
                        break

            # Use canonical node if it exists, otherwise create placeholder
            if canonical_node_id:
                target_node_id = canonical_node_id
            elif not graph.has_node(target_node_id):
                graph.add_node(
                    target_node_id,
                    kind=rel.target.kind,
                    name=rel.target.name,
                    namespace=rel.target.namespace,
                    new=False,  # Placeholder node, might be filled in later
                )

            if not graph.has_edge(node_id, target_node_id):
                graph.add_edge(
                    node_id,
                    target_node_id,
                    relationship_type=rel.relationship_type.value,
                    details=rel.details,
                )
                self.discovery_stats["edges_added"] += 1

            if target_node_id not in visited and depth > 1:
                neighbor_tasks.append(
                    self._expand_from_node(
                        graph,
                        rel.target,
                        depth - 1,
                        visited,
                        discovery_options,
                        build_options,
                        cached_graph,
                    )
                )

        if neighbor_tasks:
            await asyncio.gather(*neighbor_tasks, return_exceptions=True)

    def _get_node_id_from_identifier(self, resource_id: ResourceIdentifier) -> str:
        return f"{resource_id.kind}:{resource_id.namespace or 'cluster'}:{resource_id.name}"

    def get_permission_notices(self) -> List[str]:
        return self.permission_errors.copy()

    def get_discovery_stats(self) -> dict:
        return self.discovery_stats.copy()

    def validate_graph(self, graph: nx.DiGraph) -> dict:
        """
        Validate graph for duplicates and consistency issues.
        Returns dict with validation results and any issues found.
        """
        issues = []
        warnings = []

        # Check for duplicate resources (same kind+namespace+name)
        resource_map = {}  # (kind, namespace, name) -> [node_ids]

        for node_id, attrs in graph.nodes(data=True):
            kind = attrs.get("kind")
            name = attrs.get("name")
            namespace = attrs.get("namespace") or "cluster"

            if kind and name:
                key = (kind, namespace, name)
                if key not in resource_map:
                    resource_map[key] = []
                resource_map[key].append(node_id)

        # Report duplicates
        duplicates = {k: v for k, v in resource_map.items() if len(v) > 1}
        if duplicates:
            for (kind, namespace, name), node_ids in duplicates.items():
                issues.append(
                    {
                        "type": "duplicate_resource",
                        "kind": kind,
                        "namespace": namespace,
                        "name": name,
                        "node_ids": node_ids,
                        "message": f"Found {len(node_ids)} nodes for same resource: {kind}/{name} in {namespace}",
                    }
                )

        # Check for nodes with null attributes
        null_nodes = []
        for node_id, attrs in graph.nodes(data=True):
            if not attrs.get("kind") or not attrs.get("name"):
                null_nodes.append({"node_id": node_id, "attrs": attrs})

        if null_nodes:
            for null_node in null_nodes:
                warnings.append(
                    {
                        "type": "incomplete_node",
                        "node_id": null_node["node_id"],
                        "message": f"Node {null_node['node_id']} has null kind or name",
                    }
                )

        # Check for orphaned edges (edges to non-existent nodes shouldn't happen with NetworkX)
        # But check for edges with missing relationship types
        edges_without_type = []
        for u, v, attrs in graph.edges(data=True):
            if not attrs.get("relationship_type"):
                edges_without_type.append((u, v))

        if edges_without_type:
            warnings.append(
                {
                    "type": "missing_edge_metadata",
                    "count": len(edges_without_type),
                    "message": f"{len(edges_without_type)} edges missing relationship_type",
                }
            )

        return {
            "valid": len(issues) == 0,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "unique_resources": len(resource_map),
            "duplicate_count": len(duplicates),
            "issues": issues,
            "warnings": warnings,
        }
