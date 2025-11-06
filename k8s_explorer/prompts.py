"""MCP prompts for guided K8s workflows."""

from fastmcp.prompts.prompt import Message, PromptResult


async def create_visual_graph() -> PromptResult:
    """
    Concise guide for K8s visualization and troubleshooting workflows.
    
    Teaches: graph tool usage, depth management, combining tools, parallel calls,
    creating readable Mermaid diagrams.
    
    Returns:
        Practical workflows with tool combinations and visualization tips
    """
    return [
        Message(
            """K8s Visualization & Troubleshooting Guide

## 🎯 Common Workflows

### 1. Visualizing Architecture
**Goal**: Create readable Mermaid diagrams from K8s resources

**Graph Tool Strategy**:
```
# Specific resource (depth=2 recommended)
build_resource_graph(namespace="X", kind="Deployment", name="Y", depth=2)

# Full namespace (watch size!)
build_resource_graph(namespace="X", depth=2-3)

# Large namespaces: Multiple focused calls + combine
build_resource_graph(namespace="X", kind="Deployment", name="app1", depth=2)
build_resource_graph(namespace="X", kind="Deployment", name="app2", depth=2)
```

**Depth Guide**:
- depth=1: Immediate neighbors only
- depth=2: **Recommended** - Good context, manageable size
- depth=3+: Deep analysis (larger payloads)

**⚡ Use Parallel Calls**:
```
# Make multiple graph calls IN PARALLEL for speed
Call simultaneously:
- build_resource_graph(namespace="X", kind="Deployment", name="app1", depth=2)
- build_resource_graph(namespace="X", kind="Deployment", name="app2", depth=2)
- build_resource_graph(namespace="X", kind="Deployment", name="app3", depth=2)

Combine results → One aggregated Mermaid
Benefits: 3x faster, cache-friendly, focused data
```

**Aggregate for readability**:
- 10 pods → "App: 10 replicas" (1 node)
- Group by function (monitoring, apps, data)
- Skip obvious edges (Deployment→ReplicaSet)
- Target: 20-40 nodes max in Mermaid

### 2. Troubleshooting Failing Pods
**Workflow**:
```
1. discover_resource(kind="Pod", name="X", depth="complete")
   → Full context: status, events, dependencies

2. get_pod_logs(name="X", tail=200, previous=true)
   → Check crash logs

3. list_resources(kind="Event", namespace="X")
   → Recent cluster events

4. discover_resource(kind="Deployment", name="Y", depth="relationships")
   → Check parent resource health

5. Combine findings → Root cause
```

**⚡ Parallel optimization**:
```
# Steps 1, 2, 3 can run IN PARALLEL - no dependencies
Call simultaneously:
- discover_resource(kind="Pod", name="X", depth="complete")
- get_pod_logs(name="X", tail=200, previous=true)
- list_resources(kind="Event", namespace="X")

Then step 4 based on results
```

### 3. Impact Analysis
**"What uses this ConfigMap?"**
```
1. build_resource_graph(namespace="X", kind="ConfigMap", name="Y", depth=2)
   → See all dependent pods/deployments

2. Create Mermaid showing:
   ConfigMap → N Deployments → M Pods
```

### 4. Investigating Changes
**"What changed?"**
```
1. get_resource_changes(kind="Deployment", name="X", max_versions=5)
   → Timeline view

2. compare_resource_versions(kind="Deployment", name="X", from_revision=5, to_revision=6)
   → Detailed diff
```

## 🛠️ Tool Combinations

**Debugging**: discover_resource + get_pod_logs + list_resources(Event) [PARALLEL]
**Visualization**: build_resource_graph + aggregate + Mermaid
**Change tracking**: get_resource_changes + compare_resource_versions
**Exploration**: list_resources + get_resource + discover_resource

## ⚡ Parallel Call Strategy

**When to parallelize**:
- ✅ Multiple independent graph queries
- ✅ Multiple resource lookups (different kinds)
- ✅ Logs + events + discovery (no dependencies)
- ✅ Different namespaces/contexts

**When NOT to parallelize**:
- ❌ When one result informs the next query
- ❌ When debugging dependencies between calls

**Example - Full namespace analysis**:
```
# Parallel: Get multiple resource types at once
Call simultaneously:
- list_resources(kind="Deployment", namespace="X")
- list_resources(kind="Service", namespace="X")
- list_resources(kind="ConfigMap", namespace="X")

Analyze results → Identify key resources

# Parallel: Deep dive into key resources
Call simultaneously:
- build_resource_graph(namespace="X", kind="Deployment", name="key-app-1", depth=2)
- build_resource_graph(namespace="X", kind="Deployment", name="key-app-2", depth=2)

Combine → Create aggregated Mermaid
```

## 🎨 Mermaid Tips

**Aggregate**:
```mermaid
subgraph "Apps (20 pods)"
    API[API: 5 replicas]
    FE[Frontend: 15 replicas]
end

subgraph "Monitoring (8 pods)"
    DD[Datadog: DaemonSet on 4 nodes]
end

API --> FE
DD -.->|observes| API
```

**Color code**: Monitoring=purple, Apps=yellow, Data=blue, Infra=gray

**Use shared_resources field**: Pods list shared configs, avoid duplicate edges

---

**Now execute**: Choose workflow, use parallel calls where possible, combine tools, create clear visualizations!""",
            role="user",
        )
    ]


async def debug_failing_pod(pod_name: str, namespace: str = "default") -> PromptResult:
    """
    Generate a comprehensive debugging workflow for a failing Kubernetes pod.

    This prompt guides you through a complete pod debugging session including:
    - Complete context discovery with all dependencies
    - Log analysis (current and previous containers)
    - Event inspection
    - Parent resource checks (Deployment/StatefulSet)
    - Configuration validation (ConfigMaps, Secrets)
    - Network connectivity checks

    Args:
        pod_name: Name of the pod to debug
        namespace: Kubernetes namespace (default: default)

    Returns:
        A structured debugging workflow
    """
    return [
        Message(
            f"""I need to debug a failing Kubernetes pod: '{pod_name}' in namespace '{namespace}'.

**Debugging Workflow**:

## Step 1: Get Complete Pod Context
```
discover_resource(
    kind="Pod",
    name="{pod_name}",
    namespace="{namespace}",
    depth="complete"
)
```
**Look for**:
- Pod status (Running, CrashLoopBackOff, Pending, etc.)
- Container statuses and restart counts
- Resource requests/limits
- Node placement
- Recent events
- Owner references (Deployment, StatefulSet, etc.)

## Step 2: Check Pod Logs (Parallel with Step 1)
```
# Current logs
get_pod_logs(
    name="{pod_name}",
    namespace="{namespace}",
    tail=200
)

# If pod is restarting, check previous container
get_pod_logs(
    name="{pod_name}",
    namespace="{namespace}",
    previous=true,
    tail=200
)
```
**Look for**:
- Error messages
- Stack traces
- OOM kills
- Configuration errors

## Step 3: Check Recent Events (Parallel with Steps 1-2)
```
list_resources(
    kind="Event",
    namespace="{namespace}"
)
```
**Filter for** events related to '{pod_name}'
**Look for**: Failed scheduling, image pull errors, probe failures

## Step 4: Validate Parent Resource
Based on owner from Step 1, check the Deployment/StatefulSet:
```
discover_resource(
    kind="Deployment",  # or StatefulSet
    name="<owner-name>",
    namespace="{namespace}",
    depth="relationships"
)
```
**Look for**:
- Desired vs actual replicas
- Recent rollout status
- Configuration changes

## Step 5: Check Dependencies
From Step 1 context, verify:
- ConfigMaps exist and are mounted correctly
- Secrets exist and are accessible
- PersistentVolumeClaims are bound
- ServiceAccount has required permissions

## Step 6: Network Validation (If Applicable)
If connectivity issues suspected:
```
# Check related services
build_resource_graph(
    namespace="{namespace}",
    kind="Service",
    name="<service-name>",
    depth=2
)
```

---

**Common Root Causes**:
1. **CrashLoopBackOff**: Check logs (Step 2) for application errors
2. **ImagePullBackOff**: Check image name, registry credentials
3. **Pending**: Check events (Step 3) for scheduling failures
4. **OOMKilled**: Check logs for memory usage, increase limits
5. **Configuration errors**: Validate ConfigMaps/Secrets from Step 1

**Next Steps**: Based on findings, recommend fixes (increase resources, fix config, etc.)""",
            role="user",
        )
    ]

