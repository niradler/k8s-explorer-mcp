import logging
import time
from threading import Lock
from typing import Optional, Tuple

import networkx as nx
from cachetools import TTLCache

from .models import GraphMergeResult, GraphMetadata

logger = logging.getLogger(__name__)


class GraphCache:

    def __init__(self, ttl: int = 300, max_size: int = 100):
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        self._lock = Lock()
        self.ttl = ttl

    def _cache_key(self, namespace: str, cluster_id: str) -> str:
        return f"{cluster_id}:{namespace}"

    def get_namespace_graph(
        self, namespace: str, cluster_id: str = "default"
    ) -> Optional[Tuple[nx.DiGraph, GraphMetadata]]:
        key = self._cache_key(namespace, cluster_id)
        with self._lock:
            cached = self._cache.get(key)
            if cached:
                logger.debug(f"Cache HIT: Namespace graph {key}")
                return cached
            logger.debug(f"Cache MISS: Namespace graph {key}")
            return None

    def merge_subgraph(
        self, namespace: str, cluster_id: str, subgraph: nx.DiGraph
    ) -> GraphMergeResult:
        key = self._cache_key(namespace, cluster_id)

        with self._lock:
            cached = self._cache.get(key)
            current_time = time.time()

            if cached:
                namespace_graph, metadata = cached
                logger.debug(f"Merging subgraph into existing namespace graph {key}")
            else:
                namespace_graph = nx.DiGraph()
                metadata = GraphMetadata(
                    namespace=namespace,
                    cluster_id=cluster_id,
                    created_at=current_time,
                    last_updated=current_time,
                    query_count=0,
                    total_nodes=0,
                    total_edges=0,
                )
                logger.debug(f"Creating new namespace graph {key}")

            existing_nodes = set(namespace_graph.nodes())
            existing_edges = set(namespace_graph.edges())

            new_nodes = set()
            updated_nodes = set()

            for node_id, node_data in subgraph.nodes(data=True):
                if node_id not in existing_nodes:
                    new_nodes.add(node_id)
                    namespace_graph.add_node(node_id, **node_data, last_seen=current_time)
                else:
                    updated_nodes.add(node_id)
                    namespace_graph.nodes[node_id].update(node_data)
                    namespace_graph.nodes[node_id]["last_seen"] = current_time

            new_edges = set()
            for source, target, edge_data in subgraph.edges(data=True):
                edge_tuple = (source, target)
                if edge_tuple not in existing_edges:
                    new_edges.add(edge_tuple)
                namespace_graph.add_edge(source, target, **edge_data)

            metadata.last_updated = current_time
            metadata.query_count += 1
            metadata.total_nodes = namespace_graph.number_of_nodes()
            metadata.total_edges = namespace_graph.number_of_edges()

            self._cache[key] = (namespace_graph, metadata)

            result = GraphMergeResult(
                new_nodes=new_nodes,
                new_edges=new_edges,
                updated_nodes=updated_nodes,
                total_nodes=metadata.total_nodes,
                total_edges=metadata.total_edges,
            )

            logger.info(
                f"Graph merge complete: {len(new_nodes)} new nodes, "
                f"{len(new_edges)} new edges, total: {metadata.total_nodes} nodes, "
                f"{metadata.total_edges} edges"
            )

            return result

    def invalidate_namespace(self, namespace: str, cluster_id: str = "default"):
        key = self._cache_key(namespace, cluster_id)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Invalidated namespace graph {key}")

    def get_metadata(self, namespace: str, cluster_id: str = "default") -> Optional[GraphMetadata]:
        cached = self.get_namespace_graph(namespace, cluster_id)
        if cached:
            _, metadata = cached
            return metadata
        return None

    def get_cache_stats(self) -> dict:
        with self._lock:
            return {
                "cached_namespaces": len(self._cache),
                "max_size": self._cache.maxsize,
                "ttl": self.ttl,
            }
