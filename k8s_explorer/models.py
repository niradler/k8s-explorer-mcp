"""Data models for K8s MCP server."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetailLevel(str, Enum):
    """Level of detail for resource responses."""

    MINIMAL = "minimal"
    SUMMARY = "summary"
    DETAILED = "detailed"
    FULL = "full"


class RelationshipType(str, Enum):
    """Types of relationships between K8s resources."""

    OWNER = "owner"
    OWNED = "owned"
    SELECTOR = "selector"
    REFERENCE = "reference"
    SERVICE = "service"
    VOLUME = "volume"
    NETWORK = "network"
    CRD = "crd"


class ResourceIdentifier(BaseModel):
    """Identifies a K8s resource."""

    kind: str
    name: str
    namespace: Optional[str] = None
    api_version: Optional[str] = "v1"

    def __hash__(self):
        return hash((self.kind, self.name, self.namespace or "", self.api_version or ""))

    def __eq__(self, other):
        if not isinstance(other, ResourceIdentifier):
            return False
        return (
            self.kind == other.kind
            and self.name == other.name
            and self.namespace == other.namespace
            and self.api_version == other.api_version
        )


class ResourceSummary(BaseModel):
    """Minimal resource information."""

    identifier: ResourceIdentifier
    status: Optional[str] = None
    created: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    owner_references: List[ResourceIdentifier] = Field(default_factory=list)


class RelatedResource(BaseModel):
    """A resource and its relationship to another resource."""

    resource: ResourceSummary
    relationship_type: RelationshipType
    relationship_details: Optional[str] = None


class ResourceTree(BaseModel):
    """A resource with its related resources in tree format."""

    resource: ResourceSummary
    children: List["ResourceTree"] = Field(default_factory=list)
    depth: int = 0


class CacheKey(BaseModel):
    """Cache key for K8s resources."""

    namespace: Optional[str]
    kind: str
    name: Optional[str] = None
    label_selector: Optional[str] = None

    def to_string(self) -> str:
        """Convert to cache key string."""
        parts = [self.kind]
        if self.namespace:
            parts.append(f"ns:{self.namespace}")
        if self.name:
            parts.append(f"name:{self.name}")
        if self.label_selector:
            parts.append(f"labels:{self.label_selector}")
        return ":".join(parts)


class ResourceFilter(BaseModel):
    """Filter configuration for resource responses."""

    detail_level: DetailLevel = DetailLevel.SUMMARY
    include_status: bool = True
    include_spec: bool = False
    include_managed_fields: bool = False
    include_annotations: bool = False
    max_annotations: int = 5
    important_annotations: List[str] = Field(
        default_factory=lambda: [
            "kubectl.kubernetes.io/last-applied-configuration",
            "deployment.kubernetes.io/revision",
        ]
    )
    max_list_items: int = 10
    status_conditions_limit: int = 5


class OperatorType(str, Enum):
    """Supported K8s operators."""

    ARGO_WORKFLOWS = "argo_workflows"
    ARGO_CD = "argo_cd"
    KNATIVE = "knative"
    AIRFLOW = "airflow"
    CERT_MANAGER = "cert_manager"
    EXTERNAL_SECRETS = "external_secrets"
    GENERIC_CRD = "generic_crd"


class CRDRelationship(BaseModel):
    """Custom Resource Definition relationship configuration."""

    crd_kind: str
    crd_group: str
    crd_version: str
    related_kinds: List[str] = Field(default_factory=list)
    label_selectors: Dict[str, str] = Field(default_factory=dict)
    special_references: List[str] = Field(default_factory=list)
