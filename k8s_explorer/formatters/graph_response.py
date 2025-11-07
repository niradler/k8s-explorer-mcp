import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


class GraphResponseFormatter:

    @staticmethod
    def format_for_llm(
        graph: nx.DiGraph,
        query_mode: str,
        namespace: str,
        cluster: str,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        depth: Optional[int] = None,
        permission_notices: Optional[List[str]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        nodes_data = []
        namespace_counts = defaultdict(int)

        for node_id, data in graph.nodes(data=True):
            ns = data.get("namespace", "cluster")
            namespace_counts[ns] += 1

            node_dict = {
                "id": node_id,
                "kind": data.get("kind"),
                "name": data.get("name"),
            }

            if "status" in data:
                node_dict["status"] = data["status"]
            if "ready" in data:
                node_dict["ready"] = data["ready"]
            if "replicas" in data:
                node_dict["replicas"] = data["replicas"]
            if "type" in data and data.get("kind") == "Service":
                node_dict["service_type"] = data["type"]

            nodes_data.append(node_dict)

        edges_data = []
        for src, tgt, data in graph.edges(data=True):
            edge_dict = {
                "source": src,
                "target": tgt,
                "relationship": data.get("relationship_type") or data.get("relationship"),
            }

            if "details" in data:
                edge_dict["details"] = data["details"]

            edges_data.append(edge_dict)

        resource_types = defaultdict(int)
        for _, data in graph.nodes(data=True):
            kind_val = data.get("kind")
            if kind_val:
                resource_types[kind_val] += 1

        query_info = {"mode": query_mode, "namespace": namespace}
        if kind:
            query_info["kind"] = kind
        if name:
            query_info["name"] = name
        if depth is not None:
            query_info["depth"] = depth

        summary = GraphResponseFormatter._create_summary(
            query_mode, namespace, kind, name, len(nodes_data), len(edges_data), resource_types
        )

        insights = GraphResponseFormatter._generate_insights(graph, query_mode, kind, name, depth)

        result = {
            "query": query_info,
            "metadata": {
                "primary_namespace": namespace,
                "cluster": cluster,
                "namespaces": dict(namespace_counts),
            },
            "counts": {
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data),
                "resource_types": dict(resource_types),
            },
            "graph": {"nodes": nodes_data[:500], "edges": edges_data[:1000]},
            "summary": summary,
            "insights": insights,
        }

        if validation_result:
            result["validation"] = validation_result
        else:
            result["validation"] = {"valid": True, "warnings": []}

        if permission_notices:
            result["permission_notice"] = "; ".join(permission_notices)

        if len(nodes_data) > 500:
            result["truncation_notice"] = (
                f"Node list truncated to 500 of {len(nodes_data)} total nodes. "
                "Use more specific starting resource for focused results."
            )

        if len(edges_data) > 1000:
            if "truncation_notice" in result:
                result["truncation_notice"] += " "
            else:
                result["truncation_notice"] = ""
            result[
                "truncation_notice"
            ] += f"Edge list truncated to 1000 of {len(edges_data)} total edges."

        return result

    @staticmethod
    def _create_summary(
        mode: str, namespace: str, kind: Optional[str], name: Optional[str], node_count: int, edge_count: int, resource_types: Dict[str, int]
    ) -> str:
        if mode == "specific_resource" and kind and name:
            summary = (
                f"Graph expanded from {kind} '{name}' in namespace '{namespace}': "
                f"discovered {node_count} resources and {edge_count} relationships. "
            )
        else:
            summary = (
                f"Full namespace graph for '{namespace}': "
                f"contains {node_count} resources and {edge_count} relationships. "
            )

        if resource_types:
            top_types = sorted(resource_types.items(), key=lambda x: -x[1])[:5]
            types_str = ", ".join(f"{count} {kind}" for kind, count in top_types)
            summary += f"Top resources: {types_str}."

        return summary

    @staticmethod
    def _generate_insights(
        graph: nx.DiGraph,
        mode: str,
        kind: Optional[str],
        name: Optional[str],
        depth: Optional[int],
    ) -> Dict[str, Any]:
        insights = {}

        if mode == "specific_resource" and kind and name:
            entry_node = None
            for node_id, data in graph.nodes(data=True):
                if data.get("kind") == kind and data.get("name") == name:
                    entry_node = node_id
                    break
            if entry_node:
                insights["entry_point"] = entry_node

        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        insights["connected_components"] = len(components)

        if depth is not None:
            insights["max_depth_reached"] = depth

        failed_resources = []
        for node_id, data in graph.nodes(data=True):
            status = data.get("status", "").lower()
            if status in ["failed", "error", "crashloopbackoff"]:
                failed_resources.append(f"{data.get('kind')}:{data.get('name')}")

        if failed_resources:
            insights["resource_health"] = f"WARNING: {len(failed_resources)} resources in failed state"
            insights["failed_resources"] = failed_resources[:10]
        else:
            insights["resource_health"] = "All resources appear healthy"

        return insights
