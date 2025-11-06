import networkx as nx

from k8s_explorer.graph.cache import GraphCache
from k8s_explorer.graph.formatter import GraphFormatter
from k8s_explorer.graph.models import BuildOptions, DiscoveryOptions, GraphMergeResult
from k8s_explorer.graph.node_identity import NodeIdentity


class TestNodeIdentity:

    def test_get_node_id_for_deployment(self):
        resource = {"kind": "Deployment", "metadata": {"name": "nginx", "namespace": "default"}}
        node_id = NodeIdentity.get_node_id(resource)
        assert node_id == "Deployment:default:nginx"

    def test_get_node_id_for_pod_with_owner(self):
        resource = {
            "kind": "Pod",
            "metadata": {
                "name": "nginx-abc123-xyz",
                "namespace": "default",
                "labels": {"pod-template-hash": "abc123"},
                "ownerReferences": [{"kind": "ReplicaSet", "name": "nginx-abc123"}],
            },
        }
        node_id = NodeIdentity.get_node_id(resource)
        assert node_id == "Pod:default:ReplicaSet-nginx-abc123:abc123"

    def test_get_node_id_for_standalone_pod(self):
        resource = {"kind": "Pod", "metadata": {"name": "debug-pod", "namespace": "default"}}
        node_id = NodeIdentity.get_node_id(resource)
        assert node_id == "Pod:default:debug-pod"

    def test_is_stable_resource(self):
        assert NodeIdentity.is_stable_resource("Deployment") == True
        assert NodeIdentity.is_stable_resource("Service") == True
        assert NodeIdentity.is_stable_resource("Pod") == False

    def test_extract_node_attributes_deployment(self):
        resource = {
            "kind": "Deployment",
            "metadata": {
                "name": "nginx",
                "namespace": "default",
                "uid": "123",
                "labels": {"app": "nginx"},
                "creationTimestamp": "2024-01-01T00:00:00Z",
            },
            "status": {"replicas": 3, "readyReplicas": 3, "availableReplicas": 3},
        }
        attrs = NodeIdentity.extract_node_attributes(resource)
        assert attrs["kind"] == "Deployment"
        assert attrs["name"] == "nginx"
        assert attrs["namespace"] == "default"
        assert attrs["status"] == "Ready"
        assert attrs["replicas"] == 3


class TestGraphCache:

    def test_create_cache(self):
        cache = GraphCache(ttl=60, max_size=10)
        assert cache.ttl == 60

    def test_get_nonexistent_graph(self):
        cache = GraphCache()
        result = cache.get_namespace_graph("default", "cluster1")
        assert result is None

    def test_merge_subgraph_new_namespace(self):
        cache = GraphCache()
        subgraph = nx.DiGraph()
        subgraph.add_node("Deployment:default:nginx", kind="Deployment", name="nginx")
        subgraph.add_node("Pod:default:nginx-abc:123", kind="Pod", name="nginx-abc-xyz")
        subgraph.add_edge(
            "Deployment:default:nginx", "Pod:default:nginx-abc:123", relationship_type="owned"
        )

        result = cache.merge_subgraph("default", "cluster1", subgraph)

        assert isinstance(result, GraphMergeResult)
        assert len(result.new_nodes) == 2
        assert len(result.new_edges) == 1
        assert result.total_nodes == 2
        assert result.total_edges == 1

    def test_merge_subgraph_existing_namespace(self):
        cache = GraphCache()

        subgraph1 = nx.DiGraph()
        subgraph1.add_node("Deployment:default:nginx", kind="Deployment", name="nginx")
        cache.merge_subgraph("default", "cluster1", subgraph1)

        subgraph2 = nx.DiGraph()
        subgraph2.add_node("Service:default:nginx", kind="Service", name="nginx")
        subgraph2.add_node("Deployment:default:nginx", kind="Deployment", name="nginx")
        subgraph2.add_edge(
            "Service:default:nginx", "Deployment:default:nginx", relationship_type="reference"
        )

        result = cache.merge_subgraph("default", "cluster1", subgraph2)

        assert len(result.new_nodes) == 1
        assert len(result.updated_nodes) == 1
        assert result.total_nodes == 2
        assert result.total_edges == 1

    def test_invalidate_namespace(self):
        cache = GraphCache()
        subgraph = nx.DiGraph()
        subgraph.add_node("Deployment:default:nginx", kind="Deployment")
        cache.merge_subgraph("default", "cluster1", subgraph)

        assert cache.get_namespace_graph("default", "cluster1") is not None

        cache.invalidate_namespace("default", "cluster1")

        assert cache.get_namespace_graph("default", "cluster1") is None


class TestGraphFormatter:

    def test_to_llm_dict(self):
        graph = nx.DiGraph()
        graph.add_node(
            "Deployment:default:nginx",
            kind="Deployment",
            name="nginx",
            namespace="default",
            status="Ready",
        )
        graph.add_node(
            "Pod:default:nginx-abc:123",
            kind="Pod",
            name="nginx-abc-xyz",
            namespace="default",
            status="Running",
        )
        graph.add_edge(
            "Deployment:default:nginx",
            "Pod:default:nginx-abc:123",
            relationship_type="owned",
            details="Owns Pod",
        )

        query_info = {
            "mode": "specific_resource",
            "kind": "Deployment",
            "name": "nginx",
            "namespace": "default",
            "cluster": "test",
            "depth": 3,
        }

        merge_result = GraphMergeResult(
            new_nodes={"Deployment:default:nginx", "Pod:default:nginx-abc:123"},
            new_edges={("Deployment:default:nginx", "Pod:default:nginx-abc:123")},
            total_nodes=2,
            total_edges=1,
        )

        import time

        from k8s_explorer.graph.models import GraphMetadata

        metadata = GraphMetadata(
            namespace="default",
            cluster_id="test",
            created_at=time.time(),
            last_updated=time.time(),
            query_count=1,
            total_nodes=2,
            total_edges=1,
        )

        result = GraphFormatter.to_llm_dict(graph, query_info, merge_result, metadata)

        assert result["namespace"] == "default"
        assert result["node_count"] == 2
        assert result["edge_count"] == 1
        assert result["new_nodes"] == 2
        assert result["new_edges"] == 1
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert "summary" in result
        assert "cache_info" in result


class TestBuildOptions:

    def test_build_options_defaults(self):
        options = BuildOptions()
        assert options.include_rbac == True
        assert options.include_network == True
        assert options.include_crds == True
        assert options.max_nodes == 500

    def test_build_options_custom(self):
        options = BuildOptions(include_rbac=False, max_nodes=100, cluster_id="prod")
        assert options.include_rbac == False
        assert options.include_network == True
        assert options.max_nodes == 100
        assert options.cluster_id == "prod"


class TestDiscoveryOptions:

    def test_discovery_options_defaults(self):
        options = DiscoveryOptions()
        assert options.include_rbac == True
        assert options.include_network == True
        assert options.include_crds == True
