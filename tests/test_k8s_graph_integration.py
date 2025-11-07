import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
import networkx as nx

from k8s_explorer.adapters import K8sGraphClientAdapter, CRDDiscovererAdapter
from k8s_explorer.cache import GraphCache
from k8s_explorer.formatters import GraphResponseFormatter
from k8s_explorer.models import ResourceIdentifier
from k8s_graph.models import ResourceIdentifier as K8sGraphResourceIdentifier


class TestK8sGraphClientAdapter:
    
    @pytest.mark.asyncio
    async def test_adapter_get_resource(self):
        mock_client = Mock()
        mock_client.context = "test-context"
        mock_client.get_resource = AsyncMock(return_value={
            "kind": "Deployment",
            "metadata": {"name": "nginx", "namespace": "default"}
        })
        
        adapter = K8sGraphClientAdapter(mock_client)
        
        resource_id = K8sGraphResourceIdentifier(kind="Deployment", name="nginx", namespace="default")
        result = await adapter.get_resource(resource_id)
        
        assert result is not None
        assert result["kind"] == "Deployment"
        assert result["metadata"]["name"] == "nginx"
        mock_client.get_resource.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adapter_list_resources(self):
        mock_client = Mock()
        mock_client.context = "test-context"
        mock_client.list_resources = AsyncMock(return_value=[
            {"kind": "Pod", "metadata": {"name": "pod-1", "namespace": "default"}},
            {"kind": "Pod", "metadata": {"name": "pod-2", "namespace": "default"}},
        ])
        
        adapter = K8sGraphClientAdapter(mock_client)
        
        result = await adapter.list_resources(kind="Pod", namespace="default")
        
        assert len(result) == 2
        assert result[0]["metadata"]["name"] == "pod-1"
        mock_client.list_resources.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adapter_handles_errors(self):
        mock_client = Mock()
        mock_client.context = "test-context"
        mock_client.get_resource = AsyncMock(side_effect=Exception("API error"))
        
        adapter = K8sGraphClientAdapter(mock_client)
        
        resource_id = ResourceIdentifier(kind="Deployment", name="nginx", namespace="default")
        result = await adapter.get_resource(resource_id)
        
        assert result is None


class TestCRDDiscovererAdapter:
    
    @pytest.mark.asyncio
    async def test_crd_adapter_supports(self):
        mock_client = Mock()
        mock_registry = Mock()
        
        mock_handler = Mock()
        mock_handler.can_handle = Mock(return_value=True)
        mock_registry.handlers = [mock_handler]
        
        adapter = CRDDiscovererAdapter(mock_client, mock_registry)
        
        resource = {"kind": "Workflow", "apiVersion": "argoproj.io/v1alpha1"}
        assert adapter.supports(resource) is True
        mock_handler.can_handle.assert_called_once_with(resource)
    
    @pytest.mark.asyncio
    async def test_crd_adapter_discover(self):
        from k8s_explorer.relationships import ResourceRelationship
        from k8s_explorer.models import RelationshipType
        
        mock_client = Mock()
        mock_registry = Mock()
        
        mock_handler = Mock()
        mock_handler.can_handle = Mock(return_value=True)
        
        test_rel = ResourceRelationship(
            source=ResourceIdentifier(kind="Workflow", name="test", namespace="default"),
            target=ResourceIdentifier(kind="Pod", name="test-pod", namespace="default"),
            relationship_type=RelationshipType.CRD,
            details="Test relationship"
        )
        mock_handler.discover_relationships = AsyncMock(return_value=[test_rel])
        mock_registry.handlers = [mock_handler]
        mock_registry.discover_crd_relationships = AsyncMock(return_value=[test_rel])
        
        adapter = CRDDiscovererAdapter(mock_client, mock_registry)
        
        resource = {"kind": "Workflow", "apiVersion": "argoproj.io/v1alpha1"}
        relationships = await adapter.discover(resource)
        
        assert len(relationships) == 1
        source_id, target_id, rel_type, details = relationships[0]
        assert source_id.kind == "Workflow"
        assert target_id.kind == "Pod"


class TestGraphCache:
    
    def test_cache_set_and_get(self):
        cache = GraphCache(ttl=60, max_size=10)
        graph = nx.DiGraph()
        graph.add_node("Pod:default:test", kind="Pod", name="test")
        
        cache.set("cluster1", "default", graph)
        
        result = cache.get("cluster1", "default")
        assert result is not None
        assert result.number_of_nodes() == 1
    
    def test_cache_miss(self):
        cache = GraphCache()
        result = cache.get("cluster1", "nonexistent")
        assert result is None
    
    def test_cache_invalidate(self):
        cache = GraphCache()
        graph = nx.DiGraph()
        
        cache.set("cluster1", "default", graph)
        assert cache.get("cluster1", "default") is not None
        
        cache.invalidate("cluster1", "default")
        assert cache.get("cluster1", "default") is None
    
    def test_cache_metadata(self):
        cache = GraphCache()
        graph = nx.DiGraph()
        graph.add_node("Pod:default:test", kind="Pod", name="test")
        
        cache.set("cluster1", "namespace1", graph)
        
        metadata = cache.get_metadata("cluster1", "namespace1")
        assert metadata is not None
        assert metadata["node_count"] == 1
        assert "age_seconds" in metadata


class TestGraphResponseFormatter:
    
    def test_format_for_llm_basic(self):
        graph = nx.DiGraph()
        graph.add_node("Deployment:default:nginx", kind="Deployment", name="nginx", namespace="default", status="Running")
        graph.add_node("Pod:default:nginx-abc", kind="Pod", name="nginx-abc", namespace="default", status="Running")
        graph.add_edge("Deployment:default:nginx", "Pod:default:nginx-abc", relationship_type="owner")
        
        result = GraphResponseFormatter.format_for_llm(
            graph,
            query_mode="specific_resource",
            namespace="default",
            cluster="test",
            kind="Deployment",
            name="nginx",
            depth=2
        )
        
        assert result["query"]["mode"] == "specific_resource"
        assert result["metadata"]["primary_namespace"] == "default"
        assert result["counts"]["total_nodes"] == 2
        assert result["counts"]["total_edges"] == 1
        assert len(result["graph"]["nodes"]) == 2
        assert len(result["graph"]["edges"]) == 1
        assert "summary" in result
        assert "insights" in result
    
    def test_format_with_truncation(self):
        graph = nx.DiGraph()
        
        for i in range(600):
            graph.add_node(f"Pod:default:pod-{i}", kind="Pod", name=f"pod-{i}", namespace="default")
        
        result = GraphResponseFormatter.format_for_llm(
            graph,
            query_mode="full_namespace",
            namespace="default",
            cluster="test",
            depth=2
        )
        
        assert result["counts"]["total_nodes"] == 600
        assert len(result["graph"]["nodes"]) == 500
        assert "truncation_notice" in result
        assert "500 of 600" in result["truncation_notice"]

