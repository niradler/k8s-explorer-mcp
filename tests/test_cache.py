import pytest

from k8s_explorer.cache import K8sCache
from k8s_explorer.models import CacheKey, ResourceIdentifier


class TestK8sCache:
    @pytest.fixture
    def cache(self):
        return K8sCache(resource_ttl=30, relationship_ttl=60, max_size=100)

    def test_cache_initialization(self, cache):
        assert cache is not None
        stats = cache.get_stats()
        assert stats["total_requests"] == 0
        assert stats["total_hits"] == 0

    def test_resource_cache(self, cache):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        data = {"kind": "Pod", "metadata": {"name": "test"}}

        cache.set_resource(rid, data)
        result = cache.get_resource(rid)

        assert result == data

    def test_resource_cache_miss(self, cache):
        rid = ResourceIdentifier(kind="Pod", name="missing", namespace="default")
        result = cache.get_resource(rid)
        assert result is None

    def test_list_cache(self, cache):
        key = CacheKey(namespace="default", kind="Pod")
        data = [{"metadata": {"name": "pod1"}}, {"metadata": {"name": "pod2"}}]

        cache.set_list_query(key, data)
        result = cache.get_list_query(key)

        assert result == data
        assert len(result) == 2

    def test_cache_stats(self, cache):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")

        cache.get_resource(rid)

        stats = cache.get_stats()
        assert stats["total_requests"] == 1
        assert stats["resource_misses"] == 1

        cache.set_resource(rid, {"test": "data"})
        cache.get_resource(rid)

        stats = cache.get_stats()
        assert stats["total_requests"] == 2
        assert stats["resource_hits"] == 1
        assert stats["hit_rate_percent"] == 50.0

    def test_invalidate_resource(self, cache):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        data = {"test": "data"}

        cache.set_resource(rid, data)
        assert cache.get_resource(rid) == data

        cache.invalidate_resource(rid)
        assert cache.get_resource(rid) is None

    def test_clear_all(self, cache):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        cache.set_resource(rid, {"test": "data"})

        cache.clear_all()
        assert cache.get_resource(rid) is None

        stats = cache.get_stats()
        assert stats["resource_cache_size"] == 0
