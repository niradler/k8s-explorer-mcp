"""Multi-layer caching for K8s resources."""

import logging
from threading import Lock
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

from .models import CacheKey, ResourceIdentifier

logger = logging.getLogger(__name__)


class K8sCache:
    """Multi-layer cache for Kubernetes resources."""

    def __init__(
        self,
        resource_ttl: int = 30,
        relationship_ttl: int = 60,
        list_query_ttl: int = 120,
        api_discovery_ttl: int = 300,
        max_size: int = 1000,
    ):
        """
        Initialize multi-layer cache.

        Args:
            resource_ttl: TTL for resource objects in seconds
            relationship_ttl: TTL for relationships in seconds
            list_query_ttl: TTL for list queries in seconds
            api_discovery_ttl: TTL for API discovery in seconds
            max_size: Maximum number of items per cache
        """
        # L1: Resource Object Cache
        self._resource_cache = TTLCache(maxsize=max_size, ttl=resource_ttl)
        self._resource_lock = Lock()

        # L2: Relationship Cache
        self._relationship_cache = TTLCache(maxsize=max_size, ttl=relationship_ttl)
        self._relationship_lock = Lock()

        # L3: List Query Cache
        self._list_cache = TTLCache(maxsize=max_size // 2, ttl=list_query_ttl)
        self._list_lock = Lock()

        # L4: API Resource Discovery Cache
        self._api_cache = TTLCache(maxsize=100, ttl=api_discovery_ttl)
        self._api_lock = Lock()

        # Statistics
        self._stats = {
            "resource_hits": 0,
            "resource_misses": 0,
            "relationship_hits": 0,
            "relationship_misses": 0,
            "list_hits": 0,
            "list_misses": 0,
            "api_hits": 0,
            "api_misses": 0,
        }
        self._stats_lock = Lock()

    # Resource Cache Methods
    def get_resource(self, resource_id: ResourceIdentifier) -> Optional[Dict[str, Any]]:
        """Get resource from cache."""
        key = self._resource_key(resource_id)
        with self._resource_lock:
            value = self._resource_cache.get(key)
            with self._stats_lock:
                if value is not None:
                    self._stats["resource_hits"] += 1
                    logger.debug(f"Cache HIT: Resource {key}")
                else:
                    self._stats["resource_misses"] += 1
                    logger.debug(f"Cache MISS: Resource {key}")
            return value

    def set_resource(self, resource_id: ResourceIdentifier, data: Dict[str, Any]):
        """Store resource in cache."""
        key = self._resource_key(resource_id)
        with self._resource_lock:
            self._resource_cache[key] = data
            logger.debug(f"Cache SET: Resource {key}")

    def _resource_key(self, resource_id: ResourceIdentifier) -> str:
        """Generate cache key for resource."""
        return f"{resource_id.kind}:{resource_id.namespace or 'cluster'}:{resource_id.name}"

    # Relationship Cache Methods
    def get_relationships(
        self, resource_id: ResourceIdentifier, relationship_type: str
    ) -> Optional[List[ResourceIdentifier]]:
        """Get cached relationships."""
        key = f"{self._resource_key(resource_id)}:rel:{relationship_type}"
        with self._relationship_lock:
            value = self._relationship_cache.get(key)
            with self._stats_lock:
                if value is not None:
                    self._stats["relationship_hits"] += 1
                    logger.debug(f"Cache HIT: Relationship {key}")
                else:
                    self._stats["relationship_misses"] += 1
                    logger.debug(f"Cache MISS: Relationship {key}")
            return value

    def set_relationships(
        self,
        resource_id: ResourceIdentifier,
        relationship_type: str,
        resources: List[ResourceIdentifier],
    ):
        """Store relationships in cache."""
        key = f"{self._resource_key(resource_id)}:rel:{relationship_type}"
        with self._relationship_lock:
            self._relationship_cache[key] = resources
            logger.debug(f"Cache SET: Relationship {key}")

    # List Query Cache Methods
    def get_list_query(self, cache_key: CacheKey) -> Optional[List[Dict[str, Any]]]:
        """Get cached list query result."""
        key = cache_key.to_string()
        with self._list_lock:
            value = self._list_cache.get(key)
            with self._stats_lock:
                if value is not None:
                    self._stats["list_hits"] += 1
                    logger.debug(f"Cache HIT: List query {key}")
                else:
                    self._stats["list_misses"] += 1
                    logger.debug(f"Cache MISS: List query {key}")
            return value

    def set_list_query(self, cache_key: CacheKey, data: List[Dict[str, Any]]):
        """Store list query result in cache."""
        key = cache_key.to_string()
        with self._list_lock:
            self._list_cache[key] = data
            logger.debug(f"Cache SET: List query {key}")

    # API Discovery Cache Methods
    def get_api_resources(self, api_version: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached API resources."""
        key = f"api:{api_version}"
        with self._api_lock:
            value = self._api_cache.get(key)
            with self._stats_lock:
                if value is not None:
                    self._stats["api_hits"] += 1
                else:
                    self._stats["api_misses"] += 1
            return value

    def set_api_resources(self, api_version: str, resources: List[Dict[str, Any]]):
        """Store API resources in cache."""
        key = f"api:{api_version}"
        with self._api_lock:
            self._api_cache[key] = resources

    # Cache Management Methods
    def invalidate_resource(self, resource_id: ResourceIdentifier):
        """Invalidate a specific resource and its relationships."""
        key = self._resource_key(resource_id)
        with self._resource_lock:
            self._resource_cache.pop(key, None)

        # Invalidate related relationships
        with self._relationship_lock:
            keys_to_remove = [k for k in self._relationship_cache.keys() if k.startswith(key)]
            for k in keys_to_remove:
                self._relationship_cache.pop(k, None)

        logger.info(f"Invalidated cache for {key}")

    def invalidate_namespace(self, namespace: str):
        """Invalidate all resources in a namespace."""
        with self._resource_lock:
            keys_to_remove = [
                k
                for k in self._resource_cache.keys()
                if f":{namespace}:" in k or f":{namespace}/" in k
            ]
            for k in keys_to_remove:
                self._resource_cache.pop(k, None)

        with self._list_lock:
            keys_to_remove = [k for k in self._list_cache.keys() if f"ns:{namespace}" in k]
            for k in keys_to_remove:
                self._list_cache.pop(k, None)

        logger.info(f"Invalidated cache for namespace {namespace}")

    def clear_all(self):
        """Clear all caches."""
        with self._resource_lock:
            self._resource_cache.clear()
        with self._relationship_lock:
            self._relationship_cache.clear()
        with self._list_lock:
            self._list_cache.clear()
        with self._api_lock:
            self._api_cache.clear()
        logger.info("Cleared all caches")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._stats_lock:
            total_hits = (
                self._stats["resource_hits"]
                + self._stats["relationship_hits"]
                + self._stats["list_hits"]
                + self._stats["api_hits"]
            )
            total_requests = total_hits + (
                self._stats["resource_misses"]
                + self._stats["relationship_misses"]
                + self._stats["list_misses"]
                + self._stats["api_misses"]
            )

            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "total_requests": total_requests,
                "total_hits": total_hits,
                "hit_rate_percent": round(hit_rate, 2),
                "resource_cache_size": len(self._resource_cache),
                "relationship_cache_size": len(self._relationship_cache),
                "list_cache_size": len(self._list_cache),
                "api_cache_size": len(self._api_cache),
                **self._stats,
            }

    def reset_stats(self):
        """Reset cache statistics."""
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0
        logger.info("Reset cache statistics")
