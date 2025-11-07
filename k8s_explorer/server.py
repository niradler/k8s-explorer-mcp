"""
Kubernetes Explorer MCP Server - Production Version

Intelligent K8s resource exploration with relationship mapping, CRD support,
and AI-powered insights using FastMCP sampling.

Usage:
    uv run server.py
"""

import logging
from typing import Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message, PromptResult
from k8s_graph import GraphBuilder as K8sGraphBuilder, BuildOptions as K8sGraphBuildOptions
from k8s_graph.discoverers import DiscovererRegistry
from k8s_graph.validator import validate_graph

from k8s_explorer.adapters import K8sGraphClientAdapter, CRDDiscovererAdapter
from k8s_explorer.cache import K8sCache, GraphCache
from k8s_explorer.changes import DeploymentHistoryTracker, ResourceDiffer
from k8s_explorer.client import K8sClient
from k8s_explorer.filters import ResponseFilter
from k8s_explorer.formatters import GraphResponseFormatter
from k8s_explorer.models import ResourceIdentifier
from k8s_explorer.operators.crd_handlers import CRDOperatorRegistry
from k8s_explorer.relationships import RelationshipDiscovery
from k8s_explorer import prompts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "K8s Explorer Server",
    instructions="""
Kubernetes resource explorer with intelligent caching and multi-cluster support.

**Multi-Context Usage:**
- All tools accept optional `context` parameter to target specific K8s clusters
- Without `context`, uses current kubectl context
- Call `list_contexts()` first to see available clusters

**Key Features:**
- Smart pod matching (handles recreation with fuzzy matching)
- Context-aware caching (60s resources, 120s relationships)
- Cache shared across contexts for efficiency
- Permission-aware responses

**Common Workflows:**
1. List contexts → Choose context → Query resources
2. `discover_resource(kind, name, namespace, context)` - Full resource context
3. `get_pod_logs(name, namespace, context)` - Auto-handles multi-container
4. `build_resource_graph(namespace, context)` - Visualize dependencies

**Best Practices:**
- Specify context when working with multiple clusters
- Use depth="relationships" for quick queries
- Cache automatically handles context isolation
- Fuzzy matching works for Pods across contexts
""",
)

k8s_clients: Dict[str, K8sClient] = {}
relationship_discoveries: Dict[str, RelationshipDiscovery] = {}
graph_builders: Dict[str, K8sGraphBuilder] = {}
k8s_graph_adapters: Dict[str, K8sGraphClientAdapter] = {}
crd_registry: Optional[CRDOperatorRegistry] = None
response_filter: Optional[ResponseFilter] = None
graph_cache: Optional[GraphCache] = None
shared_cache: Optional[K8sCache] = None


def _ensure_initialized(context: Optional[str] = None):
    """Ensure K8s client is initialized for the given context.
    
    Args:
        context: Kubernetes context name. If None, uses current active context.
        
    Returns:
        Tuple of (k8s_client, relationship_discovery, graph_builder)
    """
    global k8s_clients, relationship_discoveries, graph_builders, k8s_graph_adapters
    global crd_registry, response_filter, graph_cache, shared_cache

    if shared_cache is None:
        logger.info("Initializing shared cache...")
        shared_cache = K8sCache(resource_ttl=60, relationship_ttl=120, list_query_ttl=180, max_size=2000)
    
    if crd_registry is None:
        crd_registry = CRDOperatorRegistry()
        response_filter = ResponseFilter(max_conditions=3, max_annotations=5)
        graph_cache = GraphCache(ttl=300, max_size=100)

    ctx_key = context or "current"
    
    if ctx_key not in k8s_clients:
        logger.info(f"Initializing Kubernetes client for context: {context or 'current'}...")
        k8s_client = K8sClient(cache=shared_cache, context=context)
        k8s_clients[ctx_key] = k8s_client
        relationship_discoveries[ctx_key] = RelationshipDiscovery(k8s_client)
        
        k8s_graph_adapter = K8sGraphClientAdapter(k8s_client)
        k8s_graph_adapters[ctx_key] = k8s_graph_adapter
        
        registry = DiscovererRegistry.get_global()
        crd_adapter = CRDDiscovererAdapter(k8s_graph_adapter, crd_registry)
        registry.register(crd_adapter)
        
        graph_builders[ctx_key] = K8sGraphBuilder(k8s_graph_adapter)
        logger.info(f"Kubernetes client and k8s-graph initialized successfully for context: {k8s_client.context}")
    
    return k8s_clients[ctx_key], relationship_discoveries[ctx_key], graph_builders[ctx_key]


@mcp.tool()
async def list_resources(
    kind: str,
    namespace: str = "default",
    labels: Optional[Dict[str, str]] = None,
    all_namespaces: bool = False,
    context: Optional[str] = None,
) -> dict:
    """
    List Kubernetes resources of any kind.

    Generic tool for listing resources. Replaces get_pods, get_deployments, get_services.

    Args:
        kind: Resource kind (Pod, Deployment, Service, ConfigMap, etc.)
        namespace: Kubernetes namespace (ignored if all_namespaces=True)
        labels: Optional label filters (e.g., {"app": "nginx"})
        all_namespaces: List across all namespaces
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        List of resources with summary info
    """
    k8s_client, _, _ = _ensure_initialized(context)

    label_selector = None
    if labels:
        label_selector = ",".join(f"{k}={v}" for k, v in labels.items())

    target_namespace = None if all_namespaces else namespace

    resources, permission_notice = await k8s_client.list_resources(
        kind=kind, namespace=target_namespace, label_selector=label_selector
    )

    result = []
    for resource in resources[:50]:
        metadata = resource.get("metadata", {})
        status = resource.get("status", {})

        summary = {
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "created": metadata.get("creationTimestamp"),
            "labels": metadata.get("labels", {}),
        }

        if kind == "Pod":
            summary["status"] = status.get("phase")
        elif kind == "Deployment":
            summary["replicas"] = resource.get("spec", {}).get("replicas")
            summary["ready"] = status.get("readyReplicas", 0)
        elif kind == "Service":
            summary["type"] = resource.get("spec", {}).get("type")
            summary["cluster_ip"] = resource.get("spec", {}).get("clusterIP")

        result.append(summary)

    response = {
        "kind": kind,
        "namespace": target_namespace or "all",
        "count": len(result),
        "resources": result,
    }

    if permission_notice:
        response["permission_notice"] = permission_notice

    return response


@mcp.tool()
async def get_resource(
    kind: str, name: str, namespace: str = "default", context: Optional[str] = None
) -> dict:
    """
    Get a specific Kubernetes resource by name.

    Generic tool that works with any resource kind.
    For Pods: Uses smart matching if exact name not found (handles pod recreation).

    Args:
        kind: Resource kind (Pod, Deployment, Service, ConfigMap, Secret, etc.)
        name: Resource name (supports fuzzy matching for Pods)
        namespace: Kubernetes namespace
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Resource details or error
    """
    k8s_client, _, _ = _ensure_initialized(context)

    resource_id = ResourceIdentifier(kind=kind, name=name, namespace=namespace)

    if kind == "Pod":
        resource, match_info = await k8s_client.get_resource_or_similar(resource_id)
    else:
        resource = await k8s_client.get_resource(resource_id)
        match_info = None

    if not resource:
        return {"error": f"Resource not found: {kind}/{name} in namespace {namespace}"}

    metadata = resource.get("metadata", {})
    result = {
        "kind": resource.get("kind"),
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "created": metadata.get("creationTimestamp"),
        "labels": metadata.get("labels", {}),
        "annotations": metadata.get("annotations", {}),
        "status": resource.get("status", {}),
    }

    if kind == "Pod":
        spec = resource.get("spec", {})
        result["containers"] = [c.get("name") for c in spec.get("containers", [])]
        result["node"] = spec.get("nodeName")

    if match_info:
        result["match_info"] = match_info

    return result


@mcp.tool()
async def kubectl(
    args: List[str], namespace: Optional[str] = "default", context: Optional[str] = None
) -> dict:
    """
    Execute kubectl commands for operations not covered by specialized tools.

    Use this for flexibility when specialized tools don't fit.
    Examples: logs, exec, port-forward, apply, delete, etc.

    Args:
        args: kubectl arguments as list (e.g., ["get", "pods", "-o", "json"])
        namespace: Namespace to use (adds -n flag automatically)
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Command output or error
    """
    k8s_client, _, _ = _ensure_initialized(context)

    try:
        import subprocess

        cmd = ["kubectl"]
        if context:
            cmd.extend(["--context", context])
        if namespace and namespace != "all":
            cmd.extend(["-n", namespace])
        cmd.extend(args)

        logger.info(f"Executing kubectl: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {
                "error": result.stderr or "Command failed",
                "returncode": result.returncode,
                "command": " ".join(cmd),
            }

        return {"success": True, "output": result.stdout, "command": " ".join(cmd)}

    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds"}
    except Exception as e:
        logger.error(f"kubectl error: {e}")
        return {"error": str(e)}


@mcp.tool()
async def list_contexts() -> dict:
    """
    List available Kubernetes contexts and accessible namespaces.

    Permission-aware: Shows which namespaces you can actually access.
    
    Note: This tool inspects the current/default context to show accessible namespaces.
    Use the 'context' parameter in other tools to switch to a different context.

    Returns:
        Available contexts, current context, and accessible namespaces
    """
    k8s_client, _, _ = _ensure_initialized()

    try:
        from kubernetes import config as k8s_config

        contexts, active_context = k8s_config.list_kube_config_contexts()

        accessible_namespaces = await k8s_client.get_accessible_namespaces()

        return {
            "current": active_context["name"] if active_context else None,
            "available": [ctx["name"] for ctx in contexts],
            "count": len(contexts),
            "accessible_namespaces": accessible_namespaces,
            "namespace_count": len(accessible_namespaces),
            "permission_note": (
                f"You have access to {len(accessible_namespaces)} namespace(s). "
                "Responses are limited to resources you can access."
            ),
            "usage_note": "To use a different context, pass the 'context' parameter to other tools.",
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def discover_resource(
    kind: str,
    name: str,
    namespace: str = "default",
    depth: str = "complete",
    context: Optional[str] = None,
) -> dict:
    """
    Discover resource information with configurable depth.

    ONE tool for all discovery needs - just choose the depth level.

    Depth levels:
    - "relationships": Fast list of connections (owners, children, volumes, CRDs)
    - "tree": Hierarchical tree structure showing resource hierarchy
    - "complete": Full context for debugging (default) - includes management info, explanations

    Use "relationships" for speed when you just need to know what's connected.
    Use "tree" when you need to visualize parent→child hierarchy.
    Use "complete" for debugging and investigation (default).

    Args:
        kind: Resource kind (Pod, Deployment, Service, etc.)
        name: Resource name
        namespace: Kubernetes namespace
        depth: Discovery depth ("relationships" | "tree" | "complete")
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Resource discovery data based on depth level
    """
    k8s_client, relationship_discovery, _ = _ensure_initialized(context)

    if depth not in ["relationships", "tree", "complete"]:
        return {"error": f"Invalid depth: {depth}. Use 'relationships', 'tree', or 'complete'"}

    resource_id = ResourceIdentifier(kind=kind, name=name, namespace=namespace)

    try:
        if depth == "relationships":
            resource = await k8s_client.get_resource(resource_id, use_cache=True)

            if not resource:
                return {"error": f"Resource not found: {kind}/{name} in {namespace}"}

            relationships = await relationship_discovery.discover_relationships(resource)
            crd_rels = await crd_registry.discover_crd_relationships(resource, k8s_client)
            relationships.extend(crd_rels)

            filtered_resource = response_filter.filter_resource(resource, detail_level="summary")

            result = {
                "depth": "relationships",
                "resource": filtered_resource,
                "relationships": {
                    "owners": [],
                    "owned": [],
                    "selects": [],
                    "volumes": [],
                    "services": [],
                    "crds": [],
                },
                "relationship_count": len(relationships),
            }

            for rel in relationships:
                rel_data = {
                    "kind": rel.target.kind,
                    "name": rel.target.name,
                    "namespace": rel.target.namespace or "cluster",
                    "details": rel.details,
                }

                if rel.relationship_type.value == "owner":
                    result["relationships"]["owners"].append(rel_data)
                elif rel.relationship_type.value == "owned":
                    result["relationships"]["owned"].append(rel_data)
                elif rel.relationship_type.value == "selector":
                    result["relationships"]["selects"].append(rel_data)
                elif (
                    rel.relationship_type.value == "volume"
                    or rel.relationship_type.value == "reference"
                ):
                    result["relationships"]["volumes"].append(rel_data)
                elif rel.relationship_type.value == "service":
                    result["relationships"]["services"].append(rel_data)
                elif rel.relationship_type.value == "crd":
                    result["relationships"]["crds"].append(rel_data)

            return result

        elif depth == "tree":
            tree = await relationship_discovery.build_resource_tree(resource_id, max_depth=5)

            if not tree:
                return {"error": f"Resource not found: {kind}/{name} in {namespace}"}

            tree["depth"] = "tree"
            return tree

        else:
            resource = await k8s_client.get_resource(resource_id, use_cache=False)

            if not resource:
                return {"error": f"Resource not found: {kind}/{name} in {namespace}"}

            relationships = await relationship_discovery.discover_relationships(resource)
            crd_rels = await crd_registry.discover_crd_relationships(resource, k8s_client)
            relationships.extend(crd_rels)

            summary = relationship_discovery.get_resource_summary_for_llm(resource, relationships)

            configmaps_used = []
            secrets_used = []
            helm_info = None
            parent_chain = []
            children_resources = []

            for rel in relationships:
                if rel.target.kind == "ConfigMap":
                    configmaps_used.append({"name": rel.target.name, "usage": rel.details})
                elif rel.target.kind == "Secret":
                    secrets_used.append({"name": rel.target.name, "usage": rel.details})
                elif rel.target.kind == "HelmRelease":
                    helm_info = {
                        "managed_by_helm": True,
                        "release_name": rel.target.name,
                        "details": rel.details,
                    }
                elif rel.relationship_type.value == "owner":
                    parent_chain.append(
                        {
                            "kind": rel.target.kind,
                            "name": rel.target.name,
                            "controller": "controller" in rel.details.lower(),
                        }
                    )
                elif rel.relationship_type.value == "owned":
                    children_resources.append({"kind": rel.target.kind, "name": rel.target.name})

            explanation = {
                "what_is_it": f"This is a {kind} named '{name}' in namespace '{namespace}'",
                "how_managed": summary["management"],
                "dependencies": {
                    "configmaps": configmaps_used,
                    "secrets": secrets_used,
                    "parent_resources": parent_chain,
                    "child_resources": children_resources,
                },
                "helm_info": helm_info,
                "relationship_count": len(relationships),
                "dependency_summary": summary["dependency_summary"],
            }

            questions_answered = [
                f"✅ ConfigMaps used: {len(configmaps_used)}",
                f"✅ Secrets used: {len(secrets_used)}",
                f"✅ Helm managed: {'Yes' if helm_info else 'No'}",
                f"✅ Parent resources: {len(parent_chain)}",
                f"✅ Child resources: {len(children_resources)}",
                f"✅ Total relationships: {len(relationships)}",
            ]

            return {
                "depth": "complete",
                "resource": {"kind": kind, "name": name, "namespace": namespace},
                "complete_context": explanation,
                "all_relationships": summary["relationships"],
                "questions_answered": questions_answered,
                "ready_for_debugging": True,
            }

    except Exception as e:
        logger.error(f"Error discovering resource: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_pod_logs(
    name: str,
    namespace: str = "default",
    container: Optional[str] = None,
    previous: bool = False,
    tail: int = 100,
    timestamps: bool = False,
    context: Optional[str] = None,
) -> dict:
    """
    Get pod logs - optimized for LLM consumption.

    Automatically handles multi-container pods, filters output, and shows truncation info.
    Much better than raw kubectl logs for LLM analysis.

    Args:
        name: Pod name (supports fuzzy matching)
        namespace: Kubernetes namespace
        container: Container name (if pod has multiple containers, required)
        previous: Get logs from previous terminated container
        tail: Number of lines to show (default: 100, max: 1000)
        timestamps: Include timestamps in output
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Pod logs with metadata (containers, truncation info, pod status)
    """
    k8s_client, _, _ = _ensure_initialized(context)

    tail = min(tail, 1000)

    try:
        resource_id = ResourceIdentifier(kind="Pod", name=name, namespace=namespace)
        pod, match_info = await k8s_client.get_resource_or_similar(resource_id)

        if not pod:
            return {"error": f"Pod not found: {name} in namespace {namespace}"}

        metadata = pod.get("metadata", {})
        spec = pod.get("spec", {})
        status = pod.get("status", {})

        actual_name = metadata.get("name")
        containers = [c.get("name") for c in spec.get("containers", [])]

        if not container:
            if len(containers) == 1:
                container = containers[0]
            else:
                return {
                    "error": "Multi-container pod requires 'container' parameter",
                    "pod": actual_name,
                    "available_containers": containers,
                    "hint": f"Call again with container='{containers[0]}'",
                }

        if container not in containers:
            return {
                "error": f"Container '{container}' not found in pod",
                "pod": actual_name,
                "available_containers": containers,
            }

        from kubernetes import client as k8s_api_client

        core_v1 = k8s_api_client.CoreV1Api()

        logs = core_v1.read_namespaced_pod_log(
            name=actual_name,
            namespace=namespace,
            container=container,
            previous=previous,
            tail_lines=tail,
            timestamps=timestamps,
        )

        log_lines = logs.split("\n") if logs else []
        truncated = len(log_lines) >= tail

        result = {
            "pod": actual_name,
            "namespace": namespace,
            "container": container,
            "pod_status": status.get("phase"),
            "lines_returned": len(log_lines),
            "truncated": truncated,
            "logs": logs,
        }

        if match_info:
            result["match_info"] = match_info

        if truncated:
            result["note"] = f"Logs truncated to last {tail} lines. Use tail parameter to see more."

        if len(containers) > 1:
            result["other_containers"] = [c for c in containers if c != container]

        return result

    except Exception as e:
        logger.error(f"Error getting pod logs: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_resource_changes(
    kind: str,
    name: str,
    namespace: str = "default",
    max_versions: Optional[int] = 5,
    context: Optional[str] = None,
) -> dict:
    """
    Get change history for a resource showing what changed between versions.

    Shows timeline of changes with diffs. Perfect for investigating issues.
    Supports: Deployments, StatefulSets (uses K8s revision history).

    Args:
        kind: Resource kind (Deployment, StatefulSet)
        name: Resource name
        namespace: Namespace
        max_versions: Max number of versions to show (default: 5, LLM can adjust)
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Timeline of changes with diffs, not full payloads
    """
    k8s_client, _, _ = _ensure_initialized(context)

    if max_versions is None:
        max_versions = 5
    max_versions = min(max_versions, 20)

    try:
        if kind == "Deployment":
            versions = await DeploymentHistoryTracker.get_deployment_history(
                k8s_client, name, namespace, max_revisions=max_versions
            )
        elif kind == "StatefulSet":
            versions = await DeploymentHistoryTracker.get_statefulset_history(
                k8s_client, name, namespace, max_revisions=max_versions
            )
        else:
            return {
                "error": f"Change tracking not supported for {kind}. "
                f"Supported: Deployment, StatefulSet"
            }

        if not versions:
            return {
                "resource": f"{kind}/{name}",
                "namespace": namespace,
                "message": "No version history found",
                "versions_available": 0,
            }

        if len(versions) < 2:
            return {
                "resource": f"{kind}/{name}",
                "namespace": namespace,
                "message": "Need at least 2 versions to show changes",
                "versions_available": len(versions),
                "current_version": versions[0] if versions else None,
            }

        timeline = ResourceDiffer.generate_timeline(versions, max_versions)

        current_version = versions[-1]
        previous_version = versions[-2]
        latest_diff = ResourceDiffer.generate_diff(previous_version, current_version)

        return {
            "resource": f"{kind}/{name}",
            "namespace": namespace,
            "versions_available": len(versions),
            "versions_shown": len(timeline) + 1,
            "latest_changes": {
                "from_revision": previous_version.get("metadata", {})
                .get("annotations", {})
                .get("deployment.kubernetes.io/revision", "unknown"),
                "to_revision": current_version.get("metadata", {})
                .get("annotations", {})
                .get("deployment.kubernetes.io/revision", "unknown"),
                "timestamp": current_version.get("metadata", {}).get("creationTimestamp"),
                "summary": ResourceDiffer.summarize_changes(latest_diff["changes"]),
                "changes": latest_diff["changes"],
                "diff": latest_diff["diff_text"],
            },
            "timeline": timeline,
            "note": "Timeline shows most significant changes. Use max_versions to see more/less history.",
        }

    except Exception as e:
        logger.error(f"Error getting resource changes: {e}")
        return {"error": str(e)}


@mcp.tool()
async def compare_resource_versions(
    kind: str,
    name: str,
    namespace: str = "default",
    from_revision: Optional[int] = None,
    to_revision: Optional[int] = None,
    context: Optional[str] = None,
) -> dict:
    """
    Compare specific versions of a resource to see exact changes.

    More detailed than get_resource_changes. Shows field-by-field diff.

    Args:
        kind: Resource kind (Deployment, StatefulSet)
        name: Resource name
        namespace: Namespace
        from_revision: Start revision (default: previous)
        to_revision: End revision (default: current)
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Detailed comparison with field-level changes
    """
    k8s_client, _, _ = _ensure_initialized(context)

    try:
        if kind == "Deployment":
            versions = await DeploymentHistoryTracker.get_deployment_history(
                k8s_client, name, namespace
            )
        elif kind == "StatefulSet":
            versions = await DeploymentHistoryTracker.get_statefulset_history(
                k8s_client, name, namespace
            )
        else:
            return {"error": f"Comparison not supported for {kind}"}

        if not versions:
            return {"error": "No version history found"}

        if len(versions) < 2:
            return {"error": "Need at least 2 versions to compare"}

        if to_revision is None:
            to_idx = len(versions) - 1
        else:
            to_idx = min(to_revision - 1, len(versions) - 1)

        if from_revision is None:
            from_idx = to_idx - 1
        else:
            from_idx = min(from_revision - 1, len(versions) - 1)

        from_idx = max(0, from_idx)
        to_idx = max(0, to_idx)

        if from_idx == to_idx:
            return {"error": "Cannot compare same version"}

        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx

        old_version = versions[from_idx]
        new_version = versions[to_idx]

        diff = ResourceDiffer.generate_diff(old_version, new_version)

        return {
            "resource": f"{kind}/{name}",
            "namespace": namespace,
            "from_revision": from_idx + 1,
            "to_revision": to_idx + 1,
            "from_timestamp": old_version.get("metadata", {}).get("creationTimestamp"),
            "to_timestamp": new_version.get("metadata", {}).get("creationTimestamp"),
            "summary": ResourceDiffer.summarize_changes(diff["changes"]),
            "has_changes": diff["has_changes"],
            "change_count": diff["change_count"],
            "changes": diff["changes"],
            "diff": diff["diff_text"],
        }

    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        return {"error": str(e)}


async def _enrich_graph_with_status(graph: "nx.DiGraph", k8s_client, namespace: str):
    """
    Enrich graph nodes with status information for debugging.
    
    Batches API calls by resource type for efficiency.
    
    Adds status for:
    - Pods: phase (Running, Pending, Failed, etc.)
    - Deployments: ready/desired replicas
    - Services: type (ClusterIP, LoadBalancer, etc.)
    - ReplicaSets: ready/desired replicas
    """
    import networkx as nx
    from collections import defaultdict
    
    # Only enrich resource types we know the client supports and we have logic for
    ENRICHABLE_KINDS = {
        "Pod", "Deployment", "ReplicaSet", "StatefulSet", "DaemonSet",
        "Service", "ConfigMap", "Secret", "Node", "PersistentVolumeClaim"
    }
    
    nodes_by_kind = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        if "status" not in data:  # Only enrich if missing
            kind = data.get("kind")
            if kind and kind in ENRICHABLE_KINDS:
                nodes_by_kind[kind].append((node_id, data))
    
    if not nodes_by_kind:
        logger.debug("All nodes already have status information")
        return
    
    total_nodes = sum(len(nodes) for nodes in nodes_by_kind.values())
    logger.info(f"Enriching {total_nodes} nodes across {len(nodes_by_kind)} resource types with status information")
    
    try:
        for kind, nodes in nodes_by_kind.items():
            try:
                # Batch fetch all resources of this kind
                # list_resources returns (resources, permission_notice) tuple
                resources, _ = await k8s_client.list_resources(
                    kind=kind,
                    namespace=namespace
                )
                
                # Create lookup map: name -> resource
                resource_map = {}
                for resource in resources:
                    name = resource.get("metadata", {}).get("name")
                    if name:
                        resource_map[name] = resource
                
                logger.debug(f"Enriching {len(nodes)} {kind}(s): found {len(resource_map)} resources from API")
                
                # Enrich all nodes of this kind
                enriched_count = 0
                for node_id, data in nodes:
                    name = data.get("name")
                    resource = resource_map.get(name)
                    
                    if not resource:
                        continue
                    
                    enriched_count += 1
                    
                    # Add status based on resource type (use explicit graph.nodes API)
                    if kind == "Pod":
                        status = resource.get("status", {})
                        graph.nodes[node_id]["status"] = status.get("phase", "Unknown")
                        
                        # Count ready containers
                        container_statuses = status.get("containerStatuses", [])
                        ready_count = sum(1 for cs in container_statuses if cs.get("ready", False))
                        total_count = len(container_statuses)
                        graph.nodes[node_id]["ready"] = f"{ready_count}/{total_count}" if total_count > 0 else "0/0"
                    
                    elif kind in ["Deployment", "ReplicaSet", "StatefulSet", "DaemonSet"]:
                        status = resource.get("status", {})
                        spec = resource.get("spec", {})
                        
                        if kind == "Deployment":
                            ready_replicas = status.get("readyReplicas", 0)
                            desired_replicas = spec.get("replicas", 0)
                            graph.nodes[node_id]["replicas"] = f"{ready_replicas}/{desired_replicas}"
                            graph.nodes[node_id]["ready"] = ready_replicas == desired_replicas
                        elif kind == "ReplicaSet":
                            ready_replicas = status.get("readyReplicas", 0)
                            desired_replicas = status.get("replicas", 0)
                            graph.nodes[node_id]["replicas"] = f"{ready_replicas}/{desired_replicas}"
                            graph.nodes[node_id]["ready"] = ready_replicas == desired_replicas
                        elif kind == "StatefulSet":
                            ready_replicas = status.get("readyReplicas", 0)
                            desired_replicas = spec.get("replicas", 0)
                            graph.nodes[node_id]["replicas"] = f"{ready_replicas}/{desired_replicas}"
                            graph.nodes[node_id]["ready"] = ready_replicas == desired_replicas
                        elif kind == "DaemonSet":
                            desired = status.get("desiredNumberScheduled", 0)
                            ready = status.get("numberReady", 0)
                            graph.nodes[node_id]["replicas"] = f"{ready}/{desired}"
                            graph.nodes[node_id]["ready"] = ready == desired
                    
                    elif kind == "Service":
                        spec = resource.get("spec", {})
                        graph.nodes[node_id]["type"] = spec.get("type", "ClusterIP")
                        graph.nodes[node_id]["status"] = "Active"
                    
                    elif kind in ["ConfigMap", "Secret"]:
                        graph.nodes[node_id]["status"] = "Active"
                
                logger.debug(f"Enriched {enriched_count}/{len(nodes)} {kind}(s) with status")
                
            except Exception as e:
                logger.warning(f"Could not enrich {kind} nodes: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error enriching graph with status: {e}", exc_info=True)


async def _discover_reverse_dependencies(graph, k8s_client, resource_kind: str, resource_name: str, namespace: str):
    """
    Discover resources that use a specific Secret, ConfigMap, Service, or PVC.
    
    This adds reverse dependency edges to the graph for resources that weren't
    discovered through forward traversal.
    """
    import networkx as nx
    
    resource_id = f"{resource_kind}:{namespace}:{resource_name}"
    
    if resource_id not in graph:
        logger.warning(f"Resource {resource_id} not in graph, skipping reverse dependency discovery")
        return
    
    try:
        # list_resources returns (resources, permission_notice) tuple
        pods, _ = await k8s_client.list_resources(kind="Pod", namespace=namespace)
        
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name")
            pod_namespace = pod.get("metadata", {}).get("namespace", namespace)
            pod_id = f"Pod:{pod_namespace}:{pod_name}"
            
            references_resource = False
            relationship_type = "unknown"
            relationship_details = ""
            
            if resource_kind in ["Secret", "ConfigMap"]:
                spec = pod.get("spec", {})
                
                for volume in spec.get("volumes", []):
                    if resource_kind == "Secret" and volume.get("secret", {}).get("secretName") == resource_name:
                        references_resource = True
                        relationship_type = "volume"
                        relationship_details = f"Mounts {resource_kind} as volume '{volume.get('name')}'"
                        break
                    elif resource_kind == "ConfigMap" and volume.get("configMap", {}).get("name") == resource_name:
                        references_resource = True
                        relationship_type = "volume"
                        relationship_details = f"Mounts {resource_kind} as volume '{volume.get('name')}'"
                        break
                
                if not references_resource:
                    for container in spec.get("containers", []) + spec.get("initContainers", []):
                        for env in container.get("env", []):
                            value_from = env.get("valueFrom", {})
                            if resource_kind == "Secret" and value_from.get("secretKeyRef", {}).get("name") == resource_name:
                                references_resource = True
                                relationship_type = "env_var"
                                relationship_details = f"Container '{container.get('name')}' uses {resource_kind} key for env var"
                                break
                            elif resource_kind == "ConfigMap" and value_from.get("configMapKeyRef", {}).get("name") == resource_name:
                                references_resource = True
                                relationship_type = "env_var"
                                relationship_details = f"Container '{container.get('name')}' uses {resource_kind} key for env var"
                                break
                        
                        for env_from in container.get("envFrom", []):
                            if resource_kind == "Secret" and env_from.get("secretRef", {}).get("name") == resource_name:
                                references_resource = True
                                relationship_type = "env_from"
                                relationship_details = f"Container '{container.get('name')}' loads all {resource_kind} keys"
                                break
                            elif resource_kind == "ConfigMap" and env_from.get("configMapRef", {}).get("name") == resource_name:
                                references_resource = True
                                relationship_type = "env_from"
                                relationship_details = f"Container '{container.get('name')}' loads all {resource_kind} keys"
                                break
            
            if references_resource:
                if pod_id not in graph:
                    graph.add_node(pod_id, 
                                 kind="Pod", 
                                 name=pod_name, 
                                 namespace=pod_namespace,
                                 status=pod.get("status", {}).get("phase", "Unknown"))
                
                if not graph.has_edge(pod_id, resource_id):
                    graph.add_edge(pod_id, resource_id, 
                                 relationship_type=relationship_type,
                                 details=relationship_details)
                    logger.debug(f"Added reverse dependency: {pod_id} -> {resource_id} ({relationship_type})")
    
    except Exception as e:
        logger.error(f"Error discovering reverse dependencies: {e}", exc_info=True)


@mcp.tool()
async def build_resource_graph(
    namespace: str,
    kind: Optional[str] = None,
    name: Optional[str] = None,
    depth: int = 6,
    include_rbac: bool = True,
    include_network: bool = True,
    include_crds: bool = True,
    cluster_id: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """
    Build K8s resource graph for namespace with optional specific resource entry point.

    Two modes:
    1. Specific resource (kind + name): Start from resource, expand bidirectionally for 'depth' hops
    2. Full namespace (no kind/name): Build complete graph of all resources in namespace

    Results incrementally merge into cached namespace graph which grows with each query.

    Args:
        namespace: K8s namespace (required) - defines graph scope and cache boundary
        kind: Optional resource kind to start from (Deployment, Pod, Service, any CRD, etc.)
              If omitted, builds graph for entire namespace
        name: Optional resource name (required if kind specified)
        depth: Expansion depth - how many levels to recursively fetch neighbors (default: 6)
               Controls graph SIZE by limiting recursive fetches, NOT relationship visibility.
               
               depth=1: Fetch only the starting resource + its immediate neighbors
               depth=2: Fetch resource + neighbors + their neighbors (recommended for focused queries)
               depth=6: Deep exploration (default, good for complete context)
               
               Example with depth=2 starting from Deployment:
               - Fetches: Deployment (depth=2) → ReplicaSet (depth=1) → Pod (depth=0)
               - Shows edges: Deployment→RS→Pod→ConfigMap (ConfigMap NOT fetched, just referenced)
               - Result: 3 fetched resources, 4 visible in graph (ConfigMap as placeholder)
               
               Higher depth = more resources fetched = larger graphs but more complete context.
               
        include_rbac: Include RBAC relationships (ServiceAccounts, Roles, RoleBindings)
        include_network: Include NetworkPolicy relationships
        include_crds: Include CRD/Operator relationships (Airflow, ArgoCD, Helm, etc.)
        cluster_id: Optional cluster identifier for multi-cluster environments
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Graph in LLM-friendly format with merged namespace graph.

    Response Optimizations:
        - Namespace is extracted to metadata (not repeated in each node)
        - "new" items listed separately instead of boolean on each item
        - Pod Sampling: Pods from same ReplicaSet template share ONE node for efficiency
          (e.g., 100 replicas = 1 node instead of 100, reducing edges by 99x)
        - Node IDs contain full context (kind:namespace:name) for clarity

    Examples:
        Specific resource entry:
        - {"namespace": "prod", "kind": "Deployment", "name": "nginx"}
        - {"namespace": "prod", "kind": "Service", "name": "api"}
        - {"namespace": "airflow", "kind": "Workflow", "name": "etl-job"}

        Full namespace graph:
        - {"namespace": "prod"}
        - {"namespace": "default", "depth": 2}
    """
    k8s_client, _, graph_builder = _ensure_initialized(context)

    try:
        if kind and not name:
            return {"error": "name is required when kind is specified"}

        cluster = cluster_id or context or "default"
        
        cached_graph = graph_cache.get(cluster, namespace)
        if cached_graph and not (kind and name):
            cache_meta = graph_cache.get_metadata(cluster, namespace)
            logger.info(f"Using cached graph for {namespace} (age: {cache_meta['age_seconds']}s)")
            
            query_mode = "full_namespace"
            validation_result = validate_graph(cached_graph)
            
            return GraphResponseFormatter.format_for_llm(
                cached_graph,
                query_mode=query_mode,
                namespace=namespace,
                cluster=cluster,
                depth=depth,
                validation_result=validation_result,
            )

        build_options = K8sGraphBuildOptions(
            include_rbac=include_rbac,
            include_network=include_network,
            include_crds=include_crds,
            max_nodes=500,
        )

        if kind and name:
            query_mode = "specific_resource"
            from k8s_graph.models import ResourceIdentifier as K8sGraphResourceId
            resource_id = K8sGraphResourceId(kind=kind, name=name, namespace=namespace)

            logger.info(f"Building graph from {kind}/{name} in namespace {namespace}")
            graph = await graph_builder.build_from_resource(resource_id, depth, build_options)
            
            if kind in ["Secret", "ConfigMap", "Service", "PersistentVolumeClaim"]:
                logger.info(f"Discovering reverse dependencies for {kind}/{name}")
                await _discover_reverse_dependencies(graph, k8s_client, kind, name, namespace)
        else:
            query_mode = "full_namespace"
            logger.info(f"Building full namespace graph for {namespace}")
            graph = await graph_builder.build_namespace_graph(namespace, depth, build_options)
        
        # Enrich graph with status information for all nodes
        logger.info(f"Enriching graph with status information...")
        await _enrich_graph_with_status(graph, k8s_client, namespace)
        
        graph_cache.set(cluster, namespace, graph)

        validation_result = validate_graph(graph)
        
        result = GraphResponseFormatter.format_for_llm(
            graph,
            query_mode=query_mode,
            namespace=namespace,
            cluster=cluster,
            kind=kind,
            name=name,
            depth=depth,
            validation_result=validation_result,
        )

        return result

    except Exception as e:
        logger.error(f"Error building resource graph: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def query_resource_graph(
    namespace: str,
    resource_kind: Optional[str] = None,
    resource_name: Optional[str] = None,
    filter_by: Optional[str] = None,
    filter_value: Optional[str] = None,
    find_path: bool = False,
    from_resource: Optional[str] = None,
    to_resource: Optional[str] = None,
    max_depth: int = 5,
    context: Optional[str] = None
) -> dict:
    """
    Query cached resource graph for fast lookups without rebuilding.
    
    Query Modes:
    1. Filter by kind: Get all resources of specific type
       - query_resource_graph(namespace="prod", filter_by="kind", filter_value="Deployment")
    
    2. Find specific resource: Get resource with immediate connections
       - query_resource_graph(namespace="prod", resource_kind="Deployment", resource_name="nginx")
    
    3. Find dependency paths: Show how resources connect
       - query_resource_graph(namespace="prod", find_path=True, 
                              from_resource="Deployment:nginx", to_resource="Secret:db-creds")
    
    4. Filter by status: Find resources in specific state
       - query_resource_graph(namespace="prod", filter_by="status", filter_value="Failed")
    
    5. Get connected resources: See what uses a resource (multi-hop with max_depth)
       - query_resource_graph(namespace="prod", resource_kind="ConfigMap", 
                              resource_name="app-config", filter_by="connections", max_depth=6)
    
    6. Find what uses a resource (reverse dependency)
       - query_resource_graph(namespace="prod", resource_kind="Secret",
                              resource_name="db-creds", filter_by="used_by", filter_value="Pod")
    
    Args:
        namespace: Kubernetes namespace to query
        resource_kind: Kind of resource to find (Deployment, Pod, etc.)
        resource_name: Name of specific resource
        filter_by: Filter type ("kind", "status", "connections", "used_by")
        filter_value: Value to filter by (or resource kind for "used_by")
        find_path: Find dependency path between resources
        from_resource: Source resource in format "kind:name"
        to_resource: Target resource in format "kind:name"
        max_depth: Maximum hops for connection queries (default: 1)
        context: Kubernetes context name (optional)
    
    Returns:
        Filtered graph results in LLM-optimized format
    """
    k8s_client, _, _ = _ensure_initialized(context)
    
    try:
        cluster = context or "default"
        
        graph = graph_cache.get(cluster, namespace)
        if not graph:
            return {
                "error": f"No cached graph found for namespace '{namespace}'. Please run build_resource_graph first.",
                "suggestion": f"Run: build_resource_graph(namespace='{namespace}')"
            }
        
        cache_meta = graph_cache.get_metadata(cluster, namespace)
        
        import networkx as nx
        
        matched_nodes = []
        matched_edges = []
        suggestions = []
        
        if find_path and from_resource and to_resource:
            from_kind, from_name = from_resource.split(":", 1)
            to_kind, to_name = to_resource.split(":", 1)
            from_id = f"{from_kind}:{namespace}:{from_name}"
            to_id = f"{to_kind}:{namespace}:{to_name}"
            
            try:
                path = nx.shortest_path(graph, source=from_id, target=to_id)
                for node_id in path:
                    attrs = graph.nodes[node_id]
                    matched_nodes.append({
                        "id": node_id,
                        "kind": attrs.get("kind"),
                        "name": attrs.get("name"),
                        "status": attrs.get("status")
                    })
                
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i+1])
                    matched_edges.append({
                        "source": path[i],
                        "target": path[i+1],
                        "relationship": edge_data.get("relationship_type", "unknown") if edge_data else "unknown"
                    })
                
                suggestions.append(f"Path length: {len(path)} hops")
            except nx.NetworkXNoPath:
                return {
                    "query": {"namespace": namespace, "find_path": True},
                    "error": f"No path found from {from_resource} to {to_resource}"
                }
        
        elif filter_by == "kind" and filter_value:
            for node_id, attrs in graph.nodes(data=True):
                if attrs.get("kind") == filter_value:
                    matched_nodes.append({
                        "id": node_id,
                        "kind": attrs.get("kind"),
                        "name": attrs.get("name"),
                        "status": attrs.get("status"),
                        "ready": attrs.get("ready")
                    })
            
            suggestions.append(f"Show pods for these {filter_value}s")
            suggestions.append(f"Find what ConfigMaps these use")
        
        elif resource_kind and resource_name:
            resource_id = f"{resource_kind}:{namespace}:{resource_name}"
            
            if resource_id not in graph:
                return {"error": f"Resource {resource_kind}/{resource_name} not found in graph"}
            
            attrs = graph.nodes[resource_id]
            matched_nodes.append({
                "id": resource_id,
                "kind": attrs.get("kind"),
                "name": attrs.get("name"),
                "status": attrs.get("status")
            })
            
            if filter_by == "connections":
                visited = {resource_id}
                to_visit = [(resource_id, 0)]
                
                while to_visit:
                    current_id, depth = to_visit.pop(0)
                    
                    if depth >= max_depth:
                        continue
                    
                    for src, tgt, edge_data in graph.edges(current_id, data=True):
                        edge_dict = {
                            "source": src,
                            "target": tgt,
                            "relationship": edge_data.get("relationship_type", "unknown"),
                            "details": edge_data.get("details", "")
                        }
                        if edge_dict not in matched_edges:
                            matched_edges.append(edge_dict)
                        
                        if tgt not in visited:
                            visited.add(tgt)
                            tgt_attrs = graph.nodes[tgt]
                            matched_nodes.append({
                                "id": tgt,
                                "kind": tgt_attrs.get("kind"),
                                "name": tgt_attrs.get("name"),
                                "status": tgt_attrs.get("status"),
                                "depth": depth + 1
                            })
                            to_visit.append((tgt, depth + 1))
                    
                    for src, tgt, edge_data in graph.in_edges(current_id, data=True):
                        edge_dict = {
                            "source": src,
                            "target": tgt,
                            "relationship": edge_data.get("relationship_type", "unknown"),
                            "details": edge_data.get("details", "")
                        }
                        if edge_dict not in matched_edges:
                            matched_edges.append(edge_dict)
                        
                        if src not in visited:
                            visited.add(src)
                            src_attrs = graph.nodes[src]
                            matched_nodes.append({
                                "id": src,
                                "kind": src_attrs.get("kind"),
                                "name": src_attrs.get("name"),
                                "status": src_attrs.get("status"),
                                "depth": depth + 1
                            })
                            to_visit.append((src, depth + 1))
                
                suggestions.append(f"Explored {max_depth} level(s) of dependencies")
                if len(matched_nodes) > 50:
                    suggestions.append(f"Large graph ({len(matched_nodes)} nodes), consider filtering by specific resource type")
            
            elif filter_by == "used_by":
                target_kind = filter_value
                for src, tgt, edge_data in graph.in_edges(resource_id, data=True):
                    src_attrs = graph.nodes[src]
                    if target_kind and src_attrs.get("kind") != target_kind:
                        continue
                    
                    matched_edges.append({
                        "source": src,
                        "target": tgt,
                        "relationship": edge_data.get("relationship_type", "unknown"),
                        "details": edge_data.get("details", "")
                    })
                    
                    if src not in [n["id"] for n in matched_nodes]:
                        matched_nodes.append({
                            "id": src,
                            "kind": src_attrs.get("kind"),
                            "name": src_attrs.get("name"),
                            "status": src_attrs.get("status")
                        })
                
                rel_types = {}
                for edge in matched_edges:
                    rel_type = edge["relationship"]
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                
                if target_kind:
                    suggestions.append(f"Found {len(matched_nodes)-1} {target_kind}(s) using this {resource_kind}")
                else:
                    suggestions.append(f"Found {len(matched_nodes)-1} resources using this {resource_kind}")
                
                if rel_types:
                    rel_summary = ", ".join(f"{count}x {rel}" for rel, count in rel_types.items())
                    suggestions.append(f"Relationship types: {rel_summary}")
        
        elif filter_by == "status" and filter_value:
            for node_id, attrs in graph.nodes(data=True):
                if attrs.get("status") == filter_value:
                    matched_nodes.append({
                        "id": node_id,
                        "kind": attrs.get("kind"),
                        "name": attrs.get("name"),
                        "status": attrs.get("status")
                    })
            
            suggestions.append(f"Get logs for these failed resources")
        
        else:
            return {"error": "Invalid query parameters. Specify filter_by, resource_kind/name, or find_path"}
        
        summary = f"Found {len(matched_nodes)} resources"
        if filter_by == "kind" and filter_value:
            summary = f"Found {len(matched_nodes)} {filter_value}(s) in namespace '{namespace}'"
        elif find_path:
            summary = f"Path from {from_resource} to {to_resource}"
        elif filter_by == "used_by":
            target_kind = filter_value or "resources"
            summary = f"Found {len(matched_nodes)-1} {target_kind}(s) using {resource_kind}/{resource_name}"
        elif filter_by == "connections":
            summary = f"{resource_kind}/{resource_name} with {len(matched_nodes)-1} connected resources (depth: {max_depth})"
        elif resource_kind and resource_name:
            summary = f"{resource_kind}/{resource_name} with {len(matched_nodes)-1} connected resources"
        
        return {
            "query": {
                "namespace": namespace,
                "cluster": cluster,
                "filter_by": filter_by,
                "filter_value": filter_value,
                "resource_kind": resource_kind,
                "resource_name": resource_name,
                "find_path": find_path,
                "from_resource": from_resource,
                "to_resource": to_resource,
                "max_depth": max_depth
            },
            "cache_hit": True,
            "cache_age_seconds": cache_meta["age_seconds"] if cache_meta else 0,
            "results": {
                "matched_nodes": len(matched_nodes),
                "matched_edges": len(matched_edges),
                "nodes": matched_nodes[:100],
                "edges": matched_edges[:100]
            },
            "summary": summary,
            "query_suggestions": suggestions if suggestions else []
        }
    
    except Exception as e:
        logger.error(f"Error querying resource graph: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.tool()
async def analyze_resource_impact(
    namespace: str,
    resource_kind: str,
    resource_name: str,
    operation: str = "delete",
    context: Optional[str] = None
) -> dict:
    """
    Analyze the impact of deleting or updating a resource.
    
    Critical for debugging: "What will break if I delete this Secret/ConfigMap?"
    
    This tool:
    - Finds all direct dependencies (Pods using the resource)
    - Finds transitive dependencies (Deployments owning those Pods)
    - Categorizes impact severity (critical, warning, safe)
    - Provides actionable recommendations
    
    Examples:
    # Check impact of deleting a Secret
    analyze_resource_impact(namespace="prod", resource_kind="Secret", 
                           resource_name="db-credentials", operation="delete")
    
    # Check impact of updating a ConfigMap
    analyze_resource_impact(namespace="prod", resource_kind="ConfigMap",
                           resource_name="app-config", operation="update")
    
    Args:
        namespace: Kubernetes namespace
        resource_kind: Kind of resource (Secret, ConfigMap, Service, etc.)
        resource_name: Name of resource
        operation: Operation type ("delete" or "update")
        context: Kubernetes context name (optional)
    
    Returns:
        Impact analysis with severity, affected resources, and recommendations
    """
    k8s_client, _, _ = _ensure_initialized(context)
    
    try:
        cluster = context or "default"
        
        graph = graph_cache.get(cluster, namespace)
        if not graph:
            return {
                "error": f"No cached graph found for namespace '{namespace}'. Please run build_resource_graph first.",
                "suggestion": f"Run: build_resource_graph(namespace='{namespace}')"
            }
        
        resource_id = f"{resource_kind}:{namespace}:{resource_name}"
        
        if resource_id not in graph:
            return {"error": f"Resource {resource_kind}/{resource_name} not found in graph"}
        
        import networkx as nx
        
        direct_dependents = []
        transitive_dependents = []
        affected_by_kind = {}
        relationship_types = {}
        
        for src, tgt, edge_data in graph.in_edges(resource_id, data=True):
            src_attrs = graph.nodes[src]
            rel_type = edge_data.get("relationship_type", "unknown")
            
            direct_dependents.append({
                "id": src,
                "kind": src_attrs.get("kind"),
                "name": src_attrs.get("name"),
                "relationship": rel_type,
                "details": edge_data.get("details", "")
            })
            
            kind = src_attrs.get("kind")
            affected_by_kind[kind] = affected_by_kind.get(kind, 0) + 1
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            try:
                ancestors = nx.ancestors(graph, src)
                for ancestor_id in ancestors:
                    ancestor_attrs = graph.nodes[ancestor_id]
                    ancestor_kind = ancestor_attrs.get("kind")
                    if ancestor_kind in ["Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"]:
                        if ancestor_id not in [t["id"] for t in transitive_dependents]:
                            transitive_dependents.append({
                                "id": ancestor_id,
                                "kind": ancestor_kind,
                                "name": ancestor_attrs.get("name"),
                                "affects": src_attrs.get("name")
                            })
            except Exception as e:
                logger.debug(f"Could not trace ancestors for {src}: {e}")
        
        severity = "safe"
        critical_count = 0
        warning_count = 0
        
        for dep in direct_dependents:
            if dep["kind"] == "Pod":
                if dep["relationship"] in ["volume", "env_var", "env_from"]:
                    critical_count += 1
                else:
                    warning_count += 1
            elif dep["kind"] in ["Deployment", "StatefulSet", "DaemonSet"]:
                critical_count += 1
            else:
                warning_count += 1
        
        if critical_count > 0:
            severity = "critical"
        elif warning_count > 0:
            severity = "warning"
        
        recommendations = []
        if operation == "delete":
            if severity == "critical":
                recommendations.append(f"⚠️ DANGER: Deleting this {resource_kind} will immediately break {critical_count} resource(s)")
                recommendations.append("Recommended: Update dependent resources first, or create a replacement")
            elif severity == "warning":
                recommendations.append(f"⚠️ WARNING: {warning_count} resource(s) reference this {resource_kind}")
                recommendations.append("Verify these resources before deletion")
            else:
                recommendations.append(f"✅ Safe to delete - no active dependencies found")
        else:
            if severity == "critical":
                recommendations.append(f"⚠️ Updating this {resource_kind} will affect {critical_count + warning_count} resource(s)")
                recommendations.append("Consider rolling restart of Pods to pick up changes")
            else:
                recommendations.append(f"Update will affect {len(direct_dependents)} resource(s)")
        
        if "Pod" in affected_by_kind:
            pod_count = affected_by_kind["Pod"]
            recommendations.append(f"💡 {pod_count} Pod(s) will need restart to reflect changes")
        
        if transitive_dependents:
            controllers = [d for d in transitive_dependents if d["kind"] in ["Deployment", "StatefulSet", "DaemonSet"]]
            if controllers:
                recommendations.append(f"💡 Consider updating {len(controllers)} controller(s) that manage affected Pods")
        
        summary = f"Impact analysis for {operation} of {resource_kind}/{resource_name}"
        if severity == "critical":
            summary += f" - 🔴 CRITICAL: {critical_count} resources at risk"
        elif severity == "warning":
            summary += f" - 🟡 WARNING: {warning_count} resources affected"
        else:
            summary += " - 🟢 SAFE: No critical dependencies"
        
        return {
            "resource": {
                "kind": resource_kind,
                "name": resource_name,
                "namespace": namespace
            },
            "operation": operation,
            "severity": severity,
            "impact": {
                "direct_dependents": len(direct_dependents),
                "transitive_dependents": len(transitive_dependents),
                "affected_by_kind": affected_by_kind,
                "relationship_types": relationship_types
            },
            "affected_resources": {
                "direct": direct_dependents[:20],
                "transitive": transitive_dependents[:20],
                "truncated": len(direct_dependents) > 20 or len(transitive_dependents) > 20
            },
            "recommendations": recommendations,
            "summary": summary
        }
    
    except Exception as e:
        logger.error(f"Error analyzing resource impact: {e}", exc_info=True)
        return {"error": str(e)}


# Register prompts from prompts.py
mcp.prompt()(prompts.create_visual_graph)
mcp.prompt()(prompts.debug_failing_pod)


def main():
    """Entry point for the k8s-explorer-mcp command."""
    logger.info("Starting K8s Explorer MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
