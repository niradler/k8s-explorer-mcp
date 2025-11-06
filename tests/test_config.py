from k8s_explorer.config import CacheConfig, FilterConfig, K8sConfig, LogLevel


class TestCacheConfig:
    def test_default_config(self):
        config = CacheConfig()
        assert config.resource_ttl == 30
        assert config.relationship_ttl == 60
        assert config.enable_cache is True

    def test_development_config(self):
        config = CacheConfig.for_development()
        assert config.resource_ttl == 10
        assert config.relationship_ttl == 20

    def test_production_config(self):
        config = CacheConfig.for_production()
        assert config.resource_ttl == 60
        assert config.relationship_ttl == 120
        assert config.max_cache_size == 5000

    def test_disabled_config(self):
        config = CacheConfig.disabled()
        assert config.enable_cache is False


class TestFilterConfig:
    def test_default_filter_config(self):
        config = FilterConfig()
        assert config.detail_level == "summary"
        assert config.include_managed_fields is False
        assert config.include_annotations is True
        assert config.max_annotations == 5


class TestK8sConfig:
    def test_default_config(self):
        config = K8sConfig()
        assert config.log_level == LogLevel.INFO
        assert config.enable_metrics is False
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.filter, FilterConfig)

    def test_development_config(self):
        config = K8sConfig.for_development()
        assert config.log_level == LogLevel.DEBUG
        assert config.cache.resource_ttl == 10

    def test_production_config(self):
        config = K8sConfig.for_production()
        assert config.log_level == LogLevel.INFO
        assert config.enable_metrics is True
        assert config.cache.resource_ttl == 60


class TestLogLevel:
    def test_log_levels(self):
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
