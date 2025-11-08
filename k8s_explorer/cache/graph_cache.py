import logging
import threading
import time
from typing import Dict, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class GraphCache:

    def __init__(self, ttl: int = 300, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[nx.DiGraph, float, float]] = {}
        self._lock = threading.RLock()

    def _make_key(self, cluster: str, namespace: str) -> str:
        return f"{cluster}:{namespace}"

    def get(self, cluster: str, namespace: str) -> Optional[nx.DiGraph]:
        key = self._make_key(cluster, namespace)

        with self._lock:
            if key not in self._cache:
                return None

            graph, created_at, last_query_time = self._cache[key]
            current_time = time.time()

            if current_time - created_at > self.ttl:
                logger.info(f"Graph cache expired for {key}")
                del self._cache[key]
                return None

            self._cache[key] = (graph, created_at, current_time)
            logger.debug(f"Graph cache hit for {key} (age: {int(current_time - created_at)}s)")
            return graph

    def set(self, cluster: str, namespace: str, graph: nx.DiGraph) -> None:
        key = self._make_key(cluster, namespace)

        with self._lock:
            current_time = time.time()
            self._cache[key] = (graph, current_time, current_time)

            if len(self._cache) > self.max_size:
                self._evict_oldest()

            logger.info(
                f"Cached graph for {key} (nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()})"
            )

    def _evict_oldest(self) -> None:
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]
        logger.debug(f"Evicted oldest graph cache entry: {oldest_key}")

    def invalidate(self, cluster: str, namespace: str) -> None:
        key = self._make_key(cluster, namespace)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.info(f"Invalidated graph cache for {key}")

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Cleared all graph cache entries")

    def get_metadata(self, cluster: str, namespace: str) -> Optional[Dict[str, any]]:
        key = self._make_key(cluster, namespace)

        with self._lock:
            if key not in self._cache:
                return None

            graph, created_at, last_query_time = self._cache[key]
            current_time = time.time()

            return {
                "key": key,
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "created_at": created_at,
                "last_query_time": last_query_time,
                "age_seconds": int(current_time - created_at),
                "ttl_remaining": int(self.ttl - (current_time - created_at)),
            }
