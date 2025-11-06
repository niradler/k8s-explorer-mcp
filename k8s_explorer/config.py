"""Configuration system with strong defaults and easy customization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class CacheConfig:
    """Cache configuration with strong defaults."""

    resource_ttl: int = 30
    relationship_ttl: int = 60
    list_query_ttl: int = 120
    api_discovery_ttl: int = 300
    max_cache_size: int = 1000
    enable_cache: bool = True
    per_cluster_isolation: bool = True

    @classmethod
    def for_development(cls) -> "CacheConfig":
        """Configuration optimized for development."""
        return cls(
            resource_ttl=10,
            relationship_ttl=20,
            list_query_ttl=30,
            api_discovery_ttl=60,
        )

    @classmethod
    def for_production(cls) -> "CacheConfig":
        """Configuration optimized for production."""
        return cls(
            resource_ttl=60,
            relationship_ttl=120,
            list_query_ttl=180,
            api_discovery_ttl=600,
            max_cache_size=5000,
        )

    @classmethod
    def disabled(cls) -> "CacheConfig":
        """Disable all caching."""
        return cls(enable_cache=False)


@dataclass
class FilterConfig:
    """Response filtering configuration."""

    detail_level: str = "summary"
    include_managed_fields: bool = False
    include_annotations: bool = True
    max_annotations: int = 5
    important_annotations: List[str] = field(
        default_factory=lambda: [
            "kubectl.kubernetes.io/last-applied-configuration",
            "deployment.kubernetes.io/revision",
        ]
    )
    max_list_items: int = 50
    status_conditions_limit: int = 5
    max_response_size_kb: int = 50


@dataclass
class DiscoveryConfig:
    """Resource discovery configuration."""

    max_depth: int = 3
    follow_owner_refs: bool = True
    follow_label_selectors: bool = True
    discover_volume_mounts: bool = True
    discover_service_endpoints: bool = True
    enable_crd_discovery: bool = True
    operator_handlers: List[str] = field(
        default_factory=lambda: [
            "argo_workflows",
            "knative",
            "cert_manager",
        ]
    )


@dataclass
class ClusterConfig:
    """Cluster connection configuration."""

    kubeconfig_path: Optional[str] = None
    context: Optional[str] = None
    namespace: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    verify_ssl: bool = True
    in_cluster: bool = False


@dataclass
class K8sConfig:
    """Main configuration class for K8s MCP."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    log_level: LogLevel = LogLevel.INFO
    enable_metrics: bool = False

    @classmethod
    def for_development(cls) -> "K8sConfig":
        """Configuration optimized for development."""
        return cls(
            cache=CacheConfig.for_development(),
            log_level=LogLevel.DEBUG,
        )

    @classmethod
    def for_production(cls) -> "K8sConfig":
        """Configuration optimized for production."""
        return cls(
            cache=CacheConfig.for_production(),
            log_level=LogLevel.INFO,
            enable_metrics=True,
        )
