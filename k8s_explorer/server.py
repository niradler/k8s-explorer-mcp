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

from k8s_explorer.cache import K8sCache
from k8s_explorer.changes import DeploymentHistoryTracker, ResourceDiffer
from k8s_explorer.client import K8sClient
from k8s_explorer.filters import ResponseFilter
from k8s_explorer.graph import BuildOptions, GraphBuilder, GraphCache, GraphFormatter
from k8s_explorer.models import ResourceIdentifier
from k8s_explorer.operators.crd_handlers import CRDOperatorRegistry
from k8s_explorer.relationships import RelationshipDiscovery

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
graph_builders: Dict[str, GraphBuilder] = {}
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
    global k8s_clients, relationship_discoveries, graph_builders
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
        graph_builders[ctx_key] = GraphBuilder(k8s_client, graph_cache)
        logger.info(f"Kubernetes client initialized successfully for context: {k8s_client.context}")
    
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
            tree = await relationship_discovery.build_resource_tree(resource_id, max_depth=3)

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


@mcp.tool()
async def build_resource_graph(
    namespace: str,
    kind: Optional[str] = None,
    name: Optional[str] = None,
    depth: int = 3,
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
        depth: How many relationship hops to traverse (default: 3)
               depth=1: immediate neighbors only
               depth=2: neighbors + their neighbors
               depth=3: 3 levels out (recommended for full context)
        include_rbac: Include RBAC relationships (ServiceAccounts, Roles, RoleBindings)
        include_network: Include NetworkPolicy relationships
        include_crds: Include CRD/Operator relationships (Airflow, ArgoCD, Helm, etc.)
        cluster_id: Optional cluster identifier for multi-cluster environments
        context: Kubernetes context name (optional, uses current context if not specified)

    Returns:
        Graph in LLM-friendly format with merged namespace graph.

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

        build_options = BuildOptions(
            include_rbac=include_rbac,
            include_network=include_network,
            include_crds=include_crds,
            max_nodes=500,
            cluster_id=cluster,
        )

        if kind and name:
            query_mode = "specific_resource"
            resource_id = ResourceIdentifier(kind=kind, name=name, namespace=namespace)

            logger.info(f"Building graph from {kind}/{name} in namespace {namespace}")
            subgraph = await graph_builder.build_from_resource(resource_id, depth, build_options)
        else:
            query_mode = "full_namespace"
            logger.info(f"Building full namespace graph for {namespace}")
            subgraph = await graph_builder.build_namespace_graph(namespace, depth, build_options)

        merge_result = graph_cache.merge_subgraph(namespace, cluster, subgraph)

        cached = graph_cache.get_namespace_graph(namespace, cluster)
        if cached:
            full_graph, metadata = cached
        else:
            logger.error("Failed to retrieve merged graph from cache")
            return {"error": "Failed to retrieve merged graph"}

        # Validate graph for duplicates and issues
        validation = graph_builder.validate_graph(full_graph)

        if not validation["valid"]:
            logger.warning(f"Graph validation found issues: {validation['issues']}")

        query_info = {
            "mode": query_mode,
            "namespace": namespace,
            "cluster": cluster,
            "depth": depth,
        }

        if query_mode == "specific_resource":
            query_info["kind"] = kind
            query_info["name"] = name

        permission_notices = graph_builder.get_permission_notices()

        result = GraphFormatter.to_llm_dict(
            full_graph,
            query_info,
            merge_result,
            metadata,
            permission_notices if permission_notices else None,
        )

        result["debug"]["permission_errors_count"] = len(permission_notices)
        result["debug"]["subgraph_nodes"] = subgraph.number_of_nodes()
        result["debug"]["subgraph_edges"] = subgraph.number_of_edges()
        result["debug"]["discovery_stats"] = graph_builder.get_discovery_stats()
        result["debug"]["validation"] = validation

        # Add validation issues to main response if found
        if validation["issues"]:
            result["validation_issues"] = validation["issues"]
        if validation["warnings"]:
            result["validation_warnings"] = validation["warnings"]

        return result

    except Exception as e:
        logger.error(f"Error building resource graph: {e}", exc_info=True)
        return {"error": str(e)}


@mcp.prompt()
async def debug_failing_pod(pod_name: str, namespace: str = "default") -> PromptResult:
    """
    Generate a comprehensive debugging workflow for a failing Kubernetes pod.

    This prompt guides you through a complete pod debugging session including:
    - Complete context discovery with all dependencies
    - ConfigMaps and Secrets analysis
    - Recent logs retrieval
    - Change history to identify what triggered the failure
    - Full dependency tree visualization

    Args:
        pod_name: Name of the failing pod (supports fuzzy matching)
        namespace: Kubernetes namespace where the pod is located

    Returns:
        A structured debugging workflow prompt
    """
    return [
        Message(
            f"""I need to debug a failing pod in Kubernetes. Please help me investigate systematically.

**Pod Details:**
- Name: {pod_name}
- Namespace: {namespace}

**Investigation Steps:**

1. **Get Complete Context** - Use `discover_resource` with depth="complete" to understand:
   - What ConfigMaps and Secrets the pod uses
   - Parent resources (Deployment, ReplicaSet, StatefulSet)
   - Whether it's managed by Helm, ArgoCD, or other operators
   - All dependencies and relationships

2. **Check Pod Logs** - Use `get_pod_logs` with:
   - tail=200 to see recent activity
   - Note: The tool handles fuzzy matching if the pod was recreated with a new suffix
   - Automatically detects single containers or prompts for container name

3. **Investigate Recent Changes** - Use `get_resource_changes` on the parent Deployment/StatefulSet:
   - Look for configuration changes that might have caused the failure
   - Check for image updates, resource limit changes, or environment variable modifications
   - max_versions=5 to see recent history

4. **Visualize Dependencies** - Use `build_resource_graph` to:
   - See the full dependency tree
   - Identify any missing or misconfigured dependencies
   - Check if related resources are healthy

5. **Analyze and Report** - Based on the findings:
   - Identify the root cause
   - Suggest specific remediation steps
   - Highlight any configuration issues or missing resources

**Important Notes:**
- The pod name supports fuzzy matching, so if the pod was recreated with a new suffix, the tools will find it automatically
- All tools are permission-aware and will indicate if access is limited
- Focus on actionable insights and specific fixes

Please execute this investigation step by step and provide a comprehensive analysis.""",
            role="user",
        )
    ]


def main():
    """Entry point for the k8s-explorer-mcp command."""
    logger.info("Starting K8s Explorer MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
