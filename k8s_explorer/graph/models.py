from typing import Optional, Set

from pydantic import BaseModel, ConfigDict, Field


class BuildOptions(BaseModel):
    include_rbac: bool = True
    include_network: bool = True
    include_crds: bool = True
    max_nodes: int = 500
    cluster_id: Optional[str] = None


class DiscoveryOptions(BaseModel):
    include_rbac: bool = True
    include_network: bool = True
    include_crds: bool = True


class GraphMergeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_nodes: Set[str] = Field(default_factory=set)
    new_edges: Set[tuple] = Field(default_factory=set)
    updated_nodes: Set[str] = Field(default_factory=set)
    total_nodes: int = 0
    total_edges: int = 0


class GraphMetadata(BaseModel):
    namespace: str
    cluster_id: str
    created_at: float
    last_updated: float
    query_count: int = 0
    total_nodes: int = 0
    total_edges: int = 0
