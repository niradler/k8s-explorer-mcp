"""
K8s MCP - Kubernetes Resource Tree Discovery for LLMs.

A powerful library and MCP server for understanding Kubernetes resource
relationships with intelligent caching, CRD support, and optimized
responses for LLM consumption.

Quick Start:
    >>> from k8s_mcp import K8sClient, K8sCache, RelationshipDiscovery
    >>> cache = K8sCache()
    >>> client = K8sClient(cache=cache)
    >>> discovery = RelationshipDiscovery(client)
    >>> pods = await client.list_resources("Pod", namespace="default")
    >>> relationships = await discovery.discover_relationships(pods[0])
"""

from .cache import K8sCache
from .client import K8sClient
from .config import CacheConfig, FilterConfig, K8sConfig, LogLevel
from .filters import ResponseFilter
from .models import (
    CacheKey,
    CRDRelationship,
    DetailLevel,
    OperatorType,
    RelatedResource,
    RelationshipType,
    ResourceFilter,
    ResourceIdentifier,
    ResourceSummary,
    ResourceTree,
)
from .operators.crd_handlers import (
    AirflowHandler,
    ArgoCDHandler,
    ArgoWorkflowsHandler,
    CRDOperatorRegistry,
    KnativeHandler,
    OperatorHandler,
)
from .permissions import PermissionCheck, PermissionChecker
from .relationships import RelationshipDiscovery, ResourceRelationship

__version__ = "0.2.0"
__author__ = "K8s MCP Contributors"
__all__ = [
    "K8sClient",
    "K8sCache",
    "K8sConfig",
    "CacheConfig",
    "FilterConfig",
    "LogLevel",
    "RelationshipDiscovery",
    "ResourceRelationship",
    "ResponseFilter",
    "PermissionChecker",
    "PermissionCheck",
    "CRDOperatorRegistry",
    "OperatorHandler",
    "ArgoWorkflowsHandler",
    "ArgoCDHandler",
    "AirflowHandler",
    "KnativeHandler",
    "ResourceIdentifier",
    "ResourceSummary",
    "RelatedResource",
    "ResourceTree",
    "DetailLevel",
    "RelationshipType",
    "ResourceFilter",
    "CacheKey",
    "OperatorType",
    "CRDRelationship",
]
