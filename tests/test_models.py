from k8s_explorer.models import (
    CacheKey,
    DetailLevel,
    RelationshipType,
    ResourceIdentifier,
    ResourceSummary,
)


class TestResourceIdentifier:
    def test_create_identifier(self):
        rid = ResourceIdentifier(kind="Pod", name="test-pod", namespace="default")
        assert rid.kind == "Pod"
        assert rid.name == "test-pod"
        assert rid.namespace == "default"

    def test_identifier_hash(self):
        rid1 = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        rid2 = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        assert hash(rid1) == hash(rid2)

    def test_identifier_equality(self):
        rid1 = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        rid2 = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        assert rid1 == rid2

        rid3 = ResourceIdentifier(kind="Pod", name="other", namespace="default")
        assert rid1 != rid3


class TestCacheKey:
    def test_cache_key_to_string(self):
        key = CacheKey(namespace="default", kind="Pod", name="test-pod")
        result = key.to_string()
        assert "Pod" in result
        assert "ns:default" in result
        assert "name:test-pod" in result

    def test_cache_key_with_labels(self):
        key = CacheKey(namespace="default", kind="Pod", label_selector="app=nginx")
        result = key.to_string()
        assert "labels:app=nginx" in result


class TestResourceSummary:
    def test_create_summary(self):
        rid = ResourceIdentifier(kind="Pod", name="test", namespace="default")
        summary = ResourceSummary(identifier=rid, status="Running", labels={"app": "test"})
        assert summary.identifier == rid
        assert summary.status == "Running"
        assert summary.labels["app"] == "test"


class TestEnums:
    def test_detail_level(self):
        assert DetailLevel.MINIMAL == "minimal"
        assert DetailLevel.SUMMARY == "summary"
        assert DetailLevel.DETAILED == "detailed"
        assert DetailLevel.FULL == "full"

    def test_relationship_type(self):
        assert RelationshipType.OWNER == "owner"
        assert RelationshipType.OWNED == "owned"
        assert RelationshipType.SELECTOR == "selector"
