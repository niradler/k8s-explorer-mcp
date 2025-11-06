import logging
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx

from .models import GraphMergeResult, GraphMetadata

logger = logging.getLogger(__name__)


class GraphFormatter:

    @staticmethod
    def to_llm_dict(
        graph: nx.DiGraph,
        query_info: Dict[str, Any],
        merge_result: GraphMergeResult,
        metadata: GraphMetadata,
        permission_notices: List[str] = None,
        pod_shared_resources: dict = None,
    ) -> Dict[str, Any]:
        nodes = []
        namespace_counts = defaultdict(int)
        pod_shared_resources = pod_shared_resources or {}
        
        for node_id, data in graph.nodes(data=True):
            ns = data.get("namespace", "cluster")
            namespace_counts[ns] += 1
            
            node_dict = {
                "id": node_id,
                "kind": data.get("kind"),
                "name": data.get("name"),
            }
            
            if node_id in pod_shared_resources:
                node_dict["shared_resources"] = pod_shared_resources[node_id]
            
            if "status" in data:
                node_dict["status"] = data["status"]
            if "ready" in data:
                node_dict["ready"] = data["ready"]
            if "replicas" in data:
                node_dict["replicas"] = data["replicas"]
            if "type" in data and data.get("kind") == "Service":
                node_dict["service_type"] = data["type"]

            nodes.append(node_dict)

        edges = []
        for src, tgt, data in graph.edges(data=True):
            edge_dict = {
                "source": src,
                "target": tgt,
                "relationship": data.get("relationship_type"),
            }

            if "details" in data:
                edge_dict["details"] = data["details"]

            edges.append(edge_dict)

        resource_types = defaultdict(int)
        for _, data in graph.nodes(data=True):
            kind = data.get("kind")
            if kind:
                resource_types[kind] += 1

        summary = GraphFormatter._create_summary(query_info, merge_result, metadata, resource_types)

        cache_exists = metadata.query_count > 0
        
        result = {
            "query": query_info,
            "metadata": {
                "primary_namespace": metadata.namespace,
                "cluster": metadata.cluster_id,
                "namespaces": dict(namespace_counts),
                "cache_hit": cache_exists,
            },
            "counts": {
                "total_nodes": merge_result.total_nodes,
                "total_edges": merge_result.total_edges,
            },
            "resource_types": dict(resource_types),
            "nodes": nodes[:500],
            "edges": edges[:1000],
            "summary": summary,
            "cache_info": {
                "exists": cache_exists,
                "age_seconds": int(metadata.last_updated - metadata.created_at) if cache_exists else 0,
                "query_count": metadata.query_count,
                "last_updated": metadata.last_updated if cache_exists else None,
            },
        }

        if permission_notices:
            result["permission_notice"] = "; ".join(permission_notices)

        if merge_result.total_nodes > 500:
            result["truncation_notice"] = (
                f"Node list truncated to 500 of {merge_result.total_nodes} total nodes. "
                f"Use more specific starting resource for focused results."
            )

        if merge_result.total_edges > 1000:
            if "truncation_notice" not in result:
                result["truncation_notice"] = ""
            else:
                result["truncation_notice"] += " "
            result[
                "truncation_notice"
            ] += f"Edge list truncated to 1000 of {merge_result.total_edges} total edges."

        return result

    @staticmethod
    def _create_summary(
        query_info: Dict[str, Any],
        merge_result: GraphMergeResult,
        metadata: GraphMetadata,
        resource_types: Dict[str, int],
    ) -> str:
        mode = query_info.get("mode", "unknown")
        namespace = metadata.namespace

        if mode == "specific_resource":
            kind = query_info.get("kind")
            name = query_info.get("name")
            summary = (
                f"Graph expanded from {kind} '{name}' in namespace '{namespace}': "
                f"discovered {len(merge_result.new_nodes)} new resources "
                f"and {len(merge_result.new_edges)} new relationships. "
            )
        else:
            summary = (
                f"Full namespace graph for '{namespace}': "
                f"added {len(merge_result.new_nodes)} new resources "
                f"and {len(merge_result.new_edges)} new relationships. "
            )

        summary += (
            f"Total graph contains {merge_result.total_nodes} resources "
            f"({', '.join(f'{count} {kind}' for kind, count in sorted(resource_types.items(), key=lambda x: -x[1])[:5])}) "
            f"with {merge_result.total_edges} relationships. "
        )

        summary += f"This is query #{metadata.query_count} building this namespace graph."

        return summary
