import asyncio

import pytest

from k8s_explorer.cache import K8sCache
from k8s_explorer.models import ResourceIdentifier

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_pod():
    return {
        "kind": "Pod",
        "apiVersion": "v1",
        "metadata": {
            "name": "test-pod",
            "namespace": "default",
            "labels": {"app": "test", "version": "v1"},
            "creationTimestamp": "2024-01-01T00:00:00Z",
        },
        "spec": {"containers": [{"name": "main", "image": "nginx:latest"}]},
        "status": {"phase": "Running"},
    }


@pytest.fixture
def sample_deployment():
    return {
        "kind": "Deployment",
        "apiVersion": "apps/v1",
        "metadata": {"name": "test-deployment", "namespace": "default", "labels": {"app": "test"}},
        "spec": {"replicas": 3, "selector": {"matchLabels": {"app": "test"}}},
        "status": {"replicas": 3, "readyReplicas": 3, "availableReplicas": 3},
    }


@pytest.fixture
def sample_deployment_pods():
    return [
        {
            "metadata": {
                "name": "test-deployment-abc123-xyz789",
                "namespace": "default",
                "labels": {"app": "test"},
            },
            "status": {"phase": "Running"},
        },
        {
            "metadata": {
                "name": "test-deployment-def456-uvw012",
                "namespace": "default",
                "labels": {"app": "test"},
            },
            "status": {"phase": "Running"},
        },
        {
            "metadata": {
                "name": "test-deployment-ghi789-rst345",
                "namespace": "default",
                "labels": {"app": "test"},
            },
            "status": {"phase": "Running"},
        },
    ]


@pytest.fixture
def sample_statefulset_pods():
    return [
        {
            "metadata": {
                "name": "database-0",
                "namespace": "default",
                "labels": {"app": "database"},
            },
            "status": {"phase": "Running"},
        },
        {
            "metadata": {
                "name": "database-1",
                "namespace": "default",
                "labels": {"app": "database"},
            },
            "status": {"phase": "Running"},
        },
        {
            "metadata": {
                "name": "database-2",
                "namespace": "default",
                "labels": {"app": "database"},
            },
            "status": {"phase": "Running"},
        },
    ]


@pytest.fixture
def sample_cronjob_pods():
    return [
        {
            "metadata": {
                "name": "backup-1234567890-abcde",
                "namespace": "default",
                "labels": {"job-name": "backup"},
            },
            "status": {"phase": "Succeeded"},
        },
        {
            "metadata": {
                "name": "backup-9876543210-fghij",
                "namespace": "default",
                "labels": {"job-name": "backup"},
            },
            "status": {"phase": "Running"},
        },
    ]


@pytest.fixture
def resource_identifier():
    return ResourceIdentifier(kind="Pod", name="test-pod", namespace="default")


@pytest.fixture
def test_cache():
    return K8sCache(resource_ttl=30, relationship_ttl=60, max_size=100)
