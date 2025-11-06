"""Response filtering to minimize LLM token usage."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseFilter:
    """Filter and minimize K8s resource responses for LLM consumption."""

    def __init__(
        self,
        include_managed_fields: bool = False,
        max_conditions: int = 3,
        max_events: int = 5,
        max_annotations: int = 5,
    ):
        """Initialize filter with constraints."""
        self.include_managed_fields = include_managed_fields
        self.max_conditions = max_conditions
        self.max_events = max_events
        self.max_annotations = max_annotations

        self.important_annotations = {
            "kubectl.kubernetes.io/last-applied-configuration",
            "deployment.kubernetes.io/revision",
            "argocd.argoproj.io/instance",
            "argocd.argoproj.io/sync-status",
            "workflows.argoproj.io/workflow",
            "fluxcd.io/automation",
        }

    def filter_resource(
        self, resource: Dict[str, Any], detail_level: str = "summary"
    ) -> Dict[str, Any]:
        """
        Filter resource based on detail level.

        Levels:
        - minimal: Only name, kind, status
        - summary: Add labels, basic spec, key status fields
        - detailed: Add more spec fields, filtered status
        - full: Include most fields (still filter managed fields)
        """
        if detail_level == "minimal":
            return self._minimal_resource(resource)
        elif detail_level == "summary":
            return self._summary_resource(resource)
        elif detail_level == "detailed":
            return self._detailed_resource(resource)
        else:
            return self._full_resource(resource)

    def _minimal_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Absolute minimum - name, kind, status only."""
        metadata = resource.get("metadata", {})
        status = resource.get("status", {})

        result = {
            "kind": resource.get("kind"),
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
        }

        phase = status.get("phase") or status.get("state") or self._extract_pod_status(status)
        if phase:
            result["status"] = phase

        result["created"] = metadata.get("creationTimestamp")

        return result

    def _summary_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Summary level - common debugging info."""
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})
        spec = resource.get("spec", {})
        status = resource.get("status", {})

        result = {
            "kind": kind,
            "apiVersion": resource.get("apiVersion"),
            "name": metadata.get("name"),
            "namespace": metadata.get("namespace"),
            "labels": self._filter_labels(metadata.get("labels", {})),
            "annotations": self._filter_annotations(metadata.get("annotations", {})),
            "created": metadata.get("creationTimestamp"),
        }

        owner_refs = metadata.get("ownerReferences", [])
        if owner_refs:
            result["owners"] = [
                {
                    "kind": ref.get("kind"),
                    "name": ref.get("name"),
                    "controller": ref.get("controller", False),
                }
                for ref in owner_refs
            ]

        if kind == "Pod":
            result["spec"] = self._filter_pod_spec(spec)
            result["status"] = self._filter_pod_status(status)
        elif kind == "Deployment":
            result["spec"] = {
                "replicas": spec.get("replicas"),
                "selector": spec.get("selector"),
                "strategy": spec.get("strategy", {}).get("type"),
            }
            result["status"] = {
                "replicas": status.get("replicas"),
                "ready": status.get("readyReplicas", 0),
                "available": status.get("availableReplicas", 0),
                "updated": status.get("updatedReplicas", 0),
            }
        elif kind == "Service":
            result["spec"] = {
                "type": spec.get("type"),
                "clusterIP": spec.get("clusterIP"),
                "selector": spec.get("selector"),
                "ports": spec.get("ports", []),
            }
        elif kind == "ConfigMap" or kind == "Secret":
            data = resource.get("data", {})
            result["data_keys"] = list(data.keys())
        else:
            conditions = status.get("conditions", [])
            if conditions:
                result["conditions"] = self._filter_conditions(conditions)

        return result

    def _detailed_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Detailed level - more spec and status fields."""
        result = self._summary_resource(resource)

        spec = resource.get("spec", {})
        status = resource.get("status", {})

        if resource.get("kind") == "Pod":
            containers = spec.get("containers", [])
            result["spec"]["containers"] = [
                {
                    "name": c.get("name"),
                    "image": c.get("image"),
                    "ports": c.get("ports", []),
                    "resources": c.get("resources", {}),
                    "env_count": len(c.get("env", [])),
                    "volume_mounts_count": len(c.get("volumeMounts", [])),
                }
                for c in containers
            ]

            volumes = spec.get("volumes", [])
            result["spec"]["volumes"] = self._summarize_volumes(volumes)

        conditions = status.get("conditions", [])
        if conditions:
            result["conditions"] = self._filter_conditions(conditions, limit=10)

        return result

    def _full_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Full resource (still filtered)."""
        result = dict(resource)

        if not self.include_managed_fields:
            metadata = result.get("metadata", {})
            metadata.pop("managedFields", None)
            metadata.pop("selfLink", None)
            metadata.pop("uid", None)
            metadata.pop("resourceVersion", None)
            metadata.pop("generation", None)

        metadata = result.get("metadata", {})
        if "annotations" in metadata:
            metadata["annotations"] = self._filter_annotations(metadata["annotations"])

        return result

    def _filter_labels(self, labels: Dict[str, str]) -> Dict[str, str]:
        """Keep most useful labels."""
        important_prefixes = [
            "app",
            "version",
            "component",
            "environment",
            "tier",
            "dag_id",
            "task_id",
            "workflows.argoproj.io",
            "serving.knative.dev",
        ]

        filtered = {}
        for k, v in labels.items():
            if any(k.startswith(prefix) for prefix in important_prefixes) or len(filtered) < 10:
                filtered[k] = v

        return filtered

    def _filter_annotations(self, annotations: Dict[str, str]) -> Dict[str, str]:
        """Keep only important annotations."""
        filtered = {}

        for k, v in annotations.items():
            if k in self.important_annotations:
                if k == "kubectl.kubernetes.io/last-applied-configuration":
                    filtered[k] = "<truncated>"
                else:
                    filtered[k] = v
            elif len(filtered) < self.max_annotations:
                filtered[k] = v

        return filtered

    def _filter_conditions(
        self, conditions: List[Dict[str, Any]], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Filter status conditions."""
        limit = limit or self.max_conditions

        filtered = []
        for cond in conditions[:limit]:
            filtered.append(
                {
                    "type": cond.get("type"),
                    "status": cond.get("status"),
                    "reason": cond.get("reason"),
                    "message": cond.get("message", "")[:200],
                    "lastTransitionTime": cond.get("lastTransitionTime"),
                }
            )

        return filtered

    def _filter_pod_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Filter pod spec for summary."""
        containers = spec.get("containers", [])

        return {
            "nodeName": spec.get("nodeName"),
            "serviceAccountName": spec.get("serviceAccountName"),
            "restartPolicy": spec.get("restartPolicy"),
            "containers": [
                {
                    "name": c.get("name"),
                    "image": c.get("image"),
                }
                for c in containers
            ],
            "volumes_count": len(spec.get("volumes", [])),
        }

    def _filter_pod_status(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Filter pod status."""
        return {
            "phase": status.get("phase"),
            "podIP": status.get("podIP"),
            "hostIP": status.get("hostIP"),
            "startTime": status.get("startTime"),
            "conditions": self._filter_conditions(status.get("conditions", [])),
            "containerStatuses": self._filter_container_statuses(
                status.get("containerStatuses", [])
            ),
        }

    def _filter_container_statuses(self, statuses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter container statuses."""
        return [
            {
                "name": s.get("name"),
                "ready": s.get("ready"),
                "restartCount": s.get("restartCount"),
                "state": list(s.get("state", {}).keys())[0] if s.get("state") else "unknown",
            }
            for s in statuses
        ]

    def _extract_pod_status(self, status: Dict[str, Any]) -> Optional[str]:
        """Extract simple pod status."""
        phase = status.get("phase")
        if phase:
            return phase

        container_statuses = status.get("containerStatuses", [])
        if container_statuses:
            for cs in container_statuses:
                if not cs.get("ready"):
                    state = cs.get("state", {})
                    if "waiting" in state:
                        return f"Waiting: {state['waiting'].get('reason')}"
                    elif "terminated" in state:
                        return f"Terminated: {state['terminated'].get('reason')}"

        return None

    def _summarize_volumes(self, volumes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Summarize volumes."""
        summaries = []
        for vol in volumes:
            name = vol.get("name")
            vol_type = None

            if "configMap" in vol:
                vol_type = f"ConfigMap: {vol['configMap'].get('name')}"
            elif "secret" in vol:
                vol_type = f"Secret: {vol['secret'].get('secretName')}"
            elif "persistentVolumeClaim" in vol:
                vol_type = f"PVC: {vol['persistentVolumeClaim'].get('claimName')}"
            elif "emptyDir" in vol:
                vol_type = "EmptyDir"
            else:
                vol_type = list(vol.keys())[1] if len(vol) > 1 else "Unknown"

            summaries.append({"name": name, "type": vol_type})

        return summaries

    def filter_list(
        self,
        resources: List[Dict[str, Any]],
        detail_level: str = "minimal",
        limit: Optional[int] = 50,
    ) -> List[Dict[str, Any]]:
        """Filter a list of resources."""
        filtered = []

        for resource in resources[:limit]:
            filtered.append(self.filter_resource(resource, detail_level))

        return filtered
