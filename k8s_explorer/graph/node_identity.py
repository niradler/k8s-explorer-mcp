import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class NodeIdentity:

    STABLE_RESOURCES = [
        "Deployment",
        "Service",
        "ConfigMap",
        "Secret",
        "StatefulSet",
        "DaemonSet",
        "Job",
        "CronJob",
        "Ingress",
        "PersistentVolumeClaim",
        "PersistentVolume",
        "StorageClass",
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "ClusterRole",
        "ClusterRoleBinding",
        "NetworkPolicy",
        "HorizontalPodAutoscaler",
        "Namespace",
        "Node",
        "Endpoints",
    ]

    @staticmethod
    def get_node_id(resource: Dict[str, Any]) -> str:
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})
        name = metadata.get("name")
        namespace = metadata.get("namespace", "cluster")

        if kind == "ReplicaSet":
            pod_template_hash = metadata.get("labels", {}).get("pod-template-hash", "")
            owner_refs = metadata.get("ownerReferences", [])
            if pod_template_hash and owner_refs:
                owner = owner_refs[0]
                return f"ReplicaSet:{namespace}:{owner.get('name')}:{pod_template_hash}"

        return f"{kind}:{namespace}:{name}"
    
    @staticmethod
    def get_pod_template_id(resource: Dict[str, Any]) -> str:
        """Get template identifier for pods from same ReplicaSet/StatefulSet."""
        if resource.get("kind") != "Pod":
            return None
        
        metadata = resource.get("metadata", {})
        namespace = metadata.get("namespace", "cluster")
        owner_refs = metadata.get("ownerReferences", [])
        
        if owner_refs:
            owner = owner_refs[0]
            owner_kind = owner.get("kind")
            owner_name = owner.get("name")
            
            template_hash = metadata.get("labels", {}).get("pod-template-hash", "")
            if template_hash:
                return f"PodTemplate:{namespace}:{owner_kind}-{owner_name}:{template_hash}"
            
            controller_revision = metadata.get("labels", {}).get("controller-revision-hash", "")
            if controller_revision:
                return f"PodTemplate:{namespace}:{owner_kind}-{owner_name}:{controller_revision}"
        
        return None

    @staticmethod
    def get_stable_resources() -> List[str]:
        return NodeIdentity.STABLE_RESOURCES.copy()

    @staticmethod
    def is_stable_resource(kind: str) -> bool:
        return kind in NodeIdentity.STABLE_RESOURCES

    @staticmethod
    def extract_node_attributes(resource: Dict[str, Any]) -> Dict[str, Any]:
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})
        spec = resource.get("spec", {})
        status = resource.get("status", {})

        attrs = {
            "kind": kind,
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace", "cluster"),
            "uid": metadata.get("uid"),
            "labels": metadata.get("labels", {}),
            "created": metadata.get("creationTimestamp"),
        }

        if kind == "Pod":
            phase = status.get("phase")
            conditions = status.get("conditions", [])
            ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
            attrs["status"] = phase
            attrs["ready"] = ready_condition.get("status") == "True" if ready_condition else False
            attrs["node_name"] = spec.get("nodeName")

        elif kind == "Deployment":
            attrs["status"] = "Ready" if status.get("availableReplicas", 0) > 0 else "NotReady"
            attrs["replicas"] = status.get("replicas", 0)
            attrs["ready_replicas"] = status.get("readyReplicas", 0)

        elif kind == "Service":
            attrs["type"] = spec.get("type", "ClusterIP")
            attrs["cluster_ip"] = spec.get("clusterIP")

        elif kind == "StatefulSet":
            attrs["replicas"] = status.get("replicas", 0)
            attrs["ready_replicas"] = status.get("readyReplicas", 0)

        elif kind in ["ConfigMap", "Secret"]:
            attrs["status"] = "Available"

        return attrs
