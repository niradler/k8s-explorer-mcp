# 🔍 K8s Explorer MCP

> An intelligent MCP server for Kubernetes resource exploration, relationship mapping, and debugging. Understands CRDs, provides AI-powered insights, and optimizes responses for LLM consumption.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0+-blue.svg)](https://gofastmcp.com)

## ✨ Features

- 🔍 **Intelligent Resource Discovery** - Find ConfigMaps/Secrets a pod uses, detect Helm charts, trace operator management
- 🌳 **Relationship Mapping** - Complete parent-child chains, service routing, volume mounts
- 📦 **13+ CRD Operators** - Helm, ArgoCD, Airflow, Argo Workflows, Knative, FluxCD, Istio, cert-manager, Tekton, Spark, KEDA, Velero, Prometheus + AI-powered fallback for unknown CRDs
- ⚡ **Smart Caching** - 4-layer cache with 80%+ hit rate for fast repeated queries
- 🎯 **Response Filtering** - 70-90% smaller responses optimized for LLM consumption
- 🔒 **Permission-Aware** - Adapts to RBAC constraints with helpful explanations
- 🤖 **AI-Powered Insights** - Natural language explanations using FastMCP sampling
- 📝 **Built-in Prompt** - Pre-configured debugging workflow prompt
- 🔄 **Multi-Cluster Support** - Seamless switching between multiple K8s contexts

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- kubectl configured with access to a Kubernetes cluster
- Git

### Installation

```bash
# Install from PyPI (recommended)
uv pip install k8s-explorer-mcp

# Or install from source
git clone https://github.com/nirwo/k8s-explorer-mcp.git
cd k8s-explorer-mcp
uv pip install -e ".[dev]"
```

### MCP Configuration

Add to your Cursor MCP configuration (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "k8s-explorer": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/k8s-explorer-mcp",
        "run",
        "server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Replace `/path/to/k8s-explorer-mcp` with your actual project path.

### As Python Library

```python
from k8s_explorer import K8sClient, K8sCache, RelationshipDiscovery
import asyncio

async def main():
    cache = K8sCache(resource_ttl=60, max_size=2000)
    client = K8sClient(cache=cache)
    discovery = RelationshipDiscovery(client)
    
    # Get a resource
    pods = await client.list_resources(
        kind="Pod",
        namespace="default",
        label_selector="app=nginx"
    )
    
    # Discover relationships
    if pods:
        relationships = await discovery.discover_relationships(pods[0])
        for rel in relationships:
            print(f"{rel.relationship_type}: {rel.target.kind}/{rel.target.name}")
    
    # Build resource tree
    from k8s_explorer.models import ResourceIdentifier
    resource_id = ResourceIdentifier(kind="Deployment", name="nginx", namespace="default")
    tree = await discovery.build_resource_tree(resource_id, max_depth=3)
    
    stats = client.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate_percent']}%")

asyncio.run(main())
```

### As MCP Server

```bash
# Start the server
uv run server.py

# Or with make
make run
```

**Available Tools** (9 streamlined tools):

### Core Operations (4)
- `list_contexts()` - List available contexts and accessible namespaces
- `list_resources(kind, namespace, labels, all_namespaces)` - Generic list any resource type
- `get_resource(kind, name, namespace)` - Get specific resource (smart matching for Pods)
- `kubectl(args, namespace)` - Execute kubectl commands for flexibility

### Discovery (1)
- **`discover_resource(kind, name, namespace, depth="complete")`** - ONE tool for all discovery needs
  - **depth="relationships"**: Fast list of connections (owners, children, volumes, CRDs)
  - **depth="tree"**: Hierarchical tree structure showing resource hierarchy
  - **depth="complete"**: Full context for debugging (default) - includes management info, explanations
  - Replaces 3 separate tools with clear depth parameter

### Logs (1)
- **`get_pod_logs(name, namespace, container, previous, tail, timestamps)`** - Get pod logs
  - Optimized for LLM consumption
  - Automatically handles multi-container pods
  - Shows truncation info and available containers

### Change Tracking (2)
- **`get_resource_changes(kind, name, namespace, max_versions)`** - Timeline of changes
  - Shows what changed between versions
  - LLM controls depth with max_versions
  
- **`compare_resource_versions(kind, name, namespace, from_revision, to_revision)`** - Version comparison
  - Detailed field-by-field comparison

### Graph Analysis (1)
- **`build_resource_graph(namespace, kind, name, depth, include_rbac, include_network, include_crds)`** - Build complete resource graph
  - Two modes: specific resource or full namespace
  - Incremental graph building with caching
  - RBAC, Network Policy, and CRD relationship support

### Built-in Prompt (1)
- **`debug_failing_pod(pod_name, namespace)`** - Complete debugging workflow for failing pods with guided investigation steps
  
**All tools and prompts are permission-aware** and will:
- Adapt responses based on your RBAC permissions
- Include permission notices when access is limited
- Provide clear guidance on missing permissions

## 🎯 What Can We Discover?

### ✅ **Everything Your LLM Needs for K8s Operations**

**ConfigMaps & Secrets**: Automatically finds all config resources a pod uses (volume mounts, env vars, projected volumes)

**Helm Charts**: Detects which Helm chart created any resource (release name, chart version, all managed resources)

**Operators (13+)**: Identifies resources managed by Helm, ArgoCD, Argo Workflows, Airflow, Knative, FluxCD, Istio, cert-manager, Tekton, Spark, KEDA, Velero, Prometheus + AI-powered fallback for unknown CRDs

**Complete Relationships**: Parent-child chains, service routing, volume dependencies, label selectors, operator management

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [agents.md](agents.md) | Comprehensive agent guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [examples/](examples/) | Usage examples |

## 📦 Project Structure

```
k8s-explorer-mcp/
├── server.py                 # MCP server
├── k8s_explorer/             # Main package
│   ├── client.py             # Kubernetes client wrapper
│   ├── cache.py              # Multi-layer caching
│   ├── config.py             # Configuration system
│   ├── models.py             # Data models
│   └── operators/            # CRD operator support
├── examples/                 # Usage examples
├── tests/                    # Test suite
└── pyproject.toml            # Project metadata
```

## 🎯 Use Cases

### 🧠 **Smart Pod Matching (NEW)**
```bash
# Pod was recreated with different suffix? No problem!
get_resource("Pod", "myapp-deployment-abc123-old999", "default")

# Automatically finds and returns:
# {
#   "name": "myapp-deployment-def456-xyz789",
#   "match_info": {
#     "fuzzy_match_used": true,
#     "original_name": "myapp-deployment-abc123-old999",
#     "matched_name": "myapp-deployment-def456-xyz789",
#     "similarity_score": 1.0,
#     "match_reason": "exact_base_match",
#     "explanation": "Pod 'myapp-deployment-abc123-old999' not found, but found 
#                     'myapp-deployment-def456-xyz789' with same base name 
#                     'myapp-deployment'. This is likely a newer instance."
#   }
# }

# Search for pods by pattern (fuzzy matching built-in)
get_resource("Pod", "cronjob-backup", "default")

# Automatically finds similar pods with similarity scores
# Handles: Deployments, StatefulSets, Jobs, CronJobs suffixes
```

### 🐛 **Debugging Made Easy**
```bash
# Get complete context for debugging (smart matching included)
discover_resource("Pod", "my-app-xyz", "production", depth="complete")

# Returns:
# - ConfigMaps it needs (and if they exist)
# - Secrets it uses (with mount details)
# - Parent Deployment/ReplicaSet
# - Helm chart managing it
# - Complete failure context with explanations
# - Match info if pod name was fuzzy matched

# Get pod logs (with automatic container detection)
get_pod_logs("my-app-xyz", "production", tail=200)

# Returns:
# - Logs from the pod (auto-detects single container)
# - Pod status and container list
# - Truncation info
# - Match info if fuzzy matching was used
```

### 📊 **Change Tracking & Investigation (NEW)**
```bash
# What changed in the last deployment?
get_resource_changes("Deployment", "nginx-deployment", "production", max_versions=3)

# Returns:
# {
#   "latest_changes": {
#     "from_revision": "5",
#     "to_revision": "6",
#     "summary": "2 field(s) modified",
#     "changes": [
#       {
#         "field": "spec.replicas",
#         "change_type": "modified",
#         "old_value": "3",
#         "new_value": "5",
#         "delta": 2,
#         "percent_change": 66.67
#       },
#       {
#         "field": "spec.template.spec.containers[0].image",
#         "change_type": "modified",
#         "old_value": "nginx:1.21",
#         "new_value": "nginx:1.22"
#       }
#     ]
#   },
#   "timeline": [...],  # History of all changes
#   "note": "Use max_versions to control how far back to look"
# }

# Compare specific versions
compare_resource_versions("Deployment", "nginx", "prod", from_revision=3, to_revision=5)

# Get full change history (defaults to last 5 versions)
get_resource_changes("Deployment", "nginx", "prod")
# Returns: Timeline of changes with diffs
```

### 🔍 **Impact Analysis**
```bash
# What will break if I delete this ConfigMap?
discover_resource("ConfigMap", "app-config", "prod", depth="tree")

# Shows:
# - All Deployments using it
# - All ReplicaSets affected
# - All Pods that will restart

# Quick check of dependencies
discover_resource("Secret", "db-password", "prod", depth="relationships")
# Fast list of all resources using this secret
```

### 🚀 **Operator Debugging**
```bash
# Debug Airflow DAG - find related resources
list_resources("Pod", "airflow", labels={"dag_id": "etl-pipeline"})

# Debug Argo Workflow - full context
discover_resource("Workflow", "data-processing", "workflows", depth="complete")

# Debug Helm release - shows chart, version, all managed resources
discover_resource("Deployment", "nginx", "default", depth="complete")
# Returns: Helm release name, chart version, all related resources
```

### 🤖 **LLM-Powered Operations**
The LLM can now understand:
- "Show me all pods created by the nginx Helm chart"
- "What ConfigMaps does this deployment use?"
- "Why is my Airflow task failing?"
- "What will break if I update this Secret?"

All with **one tool call** and **complete context**!

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_cache.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install with dev dependencies
make install-dev

# Format code
make format

# Lint code
make lint

# Run all checks
make check
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the Kubernetes community**
