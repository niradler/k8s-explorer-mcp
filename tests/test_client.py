"""Tests for K8s client with fuzzy matching support."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from k8s_explorer.cache import K8sCache
from k8s_explorer.client import K8sClient
from k8s_explorer.models import ResourceIdentifier


@pytest.fixture
def mock_k8s_apis():
    """Mock Kubernetes API clients."""
    with (
        patch("k8s_explorer.client.client") as mock_client,
        patch("k8s_explorer.client.config") as mock_config,
    ):

        mock_config.load_kube_config = MagicMock()

        mock_core = MagicMock()
        mock_apps = MagicMock()
        mock_batch = MagicMock()
        mock_networking = MagicMock()

        mock_client.CoreV1Api.return_value = mock_core
        mock_client.AppsV1Api.return_value = mock_apps
        mock_client.BatchV1Api.return_value = mock_batch
        mock_client.NetworkingV1Api.return_value = mock_networking
        mock_client.ApiClient.return_value = MagicMock()

        yield {
            "core": mock_core,
            "apps": mock_apps,
            "batch": mock_batch,
            "networking": mock_networking,
            "config": mock_config,
        }


@pytest.fixture
def k8s_client(mock_k8s_apis):
    """Create K8sClient with mocked APIs."""
    cache = K8sCache(resource_ttl=30, relationship_ttl=60, max_size=100)
    return K8sClient(cache=cache, check_permissions=False)


class TestK8sClient:

    def test_client_initialization(self, k8s_client):
        assert k8s_client is not None
        assert k8s_client.cache is not None
        assert k8s_client.permission_checker is None

    @pytest.mark.asyncio
    async def test_get_resource_cache_hit(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        cached_data = {"kind": "Pod", "metadata": {"name": "test"}}

        k8s_client.cache.set_resource(rid, cached_data)

        result = await k8s_client.get_resource(rid, use_cache=True)
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_get_resource_cache_disabled(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        cached_data = {"kind": "Pod", "metadata": {"name": "test"}}

        k8s_client.cache.set_resource(rid, cached_data)

        mock_response = MagicMock()
        mock_response.metadata.name = "test"

        k8s_client.core_v1.read_namespaced_pod = MagicMock(return_value=mock_response)
        k8s_client.api_client.sanitize_for_serialization = MagicMock(
            return_value={"kind": "Pod", "metadata": {"name": "test"}}
        )

        result = await k8s_client.get_resource(rid, use_cache=False)
        assert result is not None


class TestFuzzyMatching:

    @pytest.mark.asyncio
    async def test_get_resource_or_similar_exact_match(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="test-pod", namespace="default")
        pod_data = {"kind": "Pod", "metadata": {"name": "test-pod"}}

        k8s_client.cache.set_resource(rid, pod_data)

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        assert result == pod_data
        assert match_info is None

    @pytest.mark.asyncio
    async def test_get_resource_or_similar_fuzzy_match(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="nginx-old123-abc", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)

        available_pods = [
            {"metadata": {"name": "nginx-abc456-def"}},
            {"metadata": {"name": "redis-xyz789-ghi"}},
        ]

        k8s_client.list_resources = AsyncMock(return_value=(available_pods, None))

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        if result:
            assert match_info is not None
            assert match_info["fuzzy_match_used"] is True
            assert "original_name" in match_info
            assert "matched_name" in match_info
            assert "similarity_score" in match_info

    @pytest.mark.asyncio
    async def test_get_resource_or_similar_no_match(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="nonexistent", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)
        k8s_client.list_resources = AsyncMock(return_value=([], None))

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        assert result is None
        assert match_info is None

    @pytest.mark.asyncio
    async def test_fuzzy_match_deployment_pattern(self, k8s_client):
        rid = ResourceIdentifier(
            kind="Pod", name="myapp-deployment-old123-xyz789", namespace="default"
        )

        k8s_client.get_resource = AsyncMock(return_value=None)

        available_pods = [
            {"metadata": {"name": "myapp-deployment-abc456-def123"}},
            {"metadata": {"name": "otherapp-deployment-ghi789-jkl456"}},
        ]

        k8s_client.list_resources = AsyncMock(return_value=(available_pods, None))

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        if result:
            assert match_info["match_reason"] == "exact_base_match"
            assert match_info["similarity_score"] == 1.0
            assert "myapp-deployment" in match_info["matched_name"]

    @pytest.mark.asyncio
    async def test_fuzzy_match_statefulset_pattern(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="database-5", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)

        available_pods = [
            {"metadata": {"name": "database-0"}},
            {"metadata": {"name": "database-1"}},
            {"metadata": {"name": "database-2"}},
        ]

        k8s_client.list_resources = AsyncMock(return_value=(available_pods, None))

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        if result:
            assert match_info["match_reason"] == "exact_base_match"
            assert "database-" in match_info["matched_name"]

    @pytest.mark.asyncio
    async def test_fuzzy_match_cronjob_pattern(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="backup-1234567890-oldxx", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)

        available_pods = [
            {"metadata": {"name": "backup-9876543210-newxx"}},
        ]

        k8s_client.list_resources = AsyncMock(return_value=(available_pods, None))

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        if result:
            assert match_info["match_reason"] == "exact_base_match"
            assert "backup-" in match_info["matched_name"]

    @pytest.mark.asyncio
    async def test_fuzzy_match_only_for_pods(self, k8s_client):
        rid = ResourceIdentifier(kind="Service", name="missing-service", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)

        result, match_info = await k8s_client.get_resource_or_similar(rid)

        assert result is None
        assert match_info is None

    @pytest.mark.asyncio
    async def test_fuzzy_match_threshold(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="completely-different-name", namespace="default")

        k8s_client.get_resource = AsyncMock(return_value=None)

        available_pods = [
            {"metadata": {"name": "nginx-deployment-abc123-def456"}},
        ]

        k8s_client.list_resources = AsyncMock(return_value=(available_pods, None))

        result, match_info = await k8s_client.get_resource_or_similar(rid, similarity_threshold=0.9)

        assert result is None
        assert match_info is None


class TestCacheStatistics:

    def test_get_cache_stats(self, k8s_client):
        stats = k8s_client.get_cache_stats()

        assert "total_requests" in stats
        assert "total_hits" in stats
        assert "hit_rate_percent" in stats

    def test_clear_cache(self, k8s_client):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        k8s_client.cache.set_resource(rid, {"test": "data"})

        k8s_client.clear_cache()

        result = k8s_client.cache.get_resource(rid)
        assert result is None
