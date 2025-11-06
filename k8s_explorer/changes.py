"""Resource change tracking and diff generation for Kubernetes resources."""

import json
import logging
from difflib import unified_diff
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResourceDiffer:

    IGNORED_FIELDS = [
        "metadata.resourceVersion",
        "metadata.generation",
        "metadata.uid",
        "metadata.selfLink",
        "metadata.managedFields",
        "status.observedGeneration",
        "status.conditions[].lastTransitionTime",
        "status.conditions[].lastUpdateTime",
    ]

    IMPORTANT_FIELDS = {
        "Deployment": [
            "spec.replicas",
            "spec.template.spec.containers",
            "spec.strategy",
            "spec.template.spec.volumes",
            "spec.selector",
        ],
        "StatefulSet": [
            "spec.replicas",
            "spec.template.spec.containers",
            "spec.volumeClaimTemplates",
            "spec.updateStrategy",
        ],
        "DaemonSet": [
            "spec.template.spec.containers",
            "spec.updateStrategy",
            "spec.template.spec.volumes",
        ],
        "ConfigMap": [
            "data",
            "binaryData",
        ],
        "Secret": [
            "data",
            "type",
        ],
        "Service": [
            "spec.type",
            "spec.ports",
            "spec.selector",
            "spec.clusterIP",
        ],
    }

    @staticmethod
    def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        value = obj

        for key in keys:
            if "[" in key and "]" in key:
                key_name = key.split("[")[0]
                if not isinstance(value, dict) or key_name not in value:
                    return None
                value = value[key_name]
            else:
                if not isinstance(value, dict) or key not in value:
                    return None
                value = value[key]

        return value

    @staticmethod
    def should_ignore_field(field_path: str) -> bool:
        for ignored in ResourceDiffer.IGNORED_FIELDS:
            if field_path.startswith(ignored) or ignored.startswith(field_path):
                return True
        return False

    @staticmethod
    def extract_important_fields(resource: Dict[str, Any]) -> Dict[str, Any]:
        kind = resource.get("kind", "Unknown")
        important = {}

        fields_to_extract = ResourceDiffer.IMPORTANT_FIELDS.get(kind, [])

        for field_path in fields_to_extract:
            value = ResourceDiffer.get_nested_value(resource, field_path)
            if value is not None:
                important[field_path] = value

        important["metadata.labels"] = resource.get("metadata", {}).get("labels", {})
        important["metadata.annotations"] = resource.get("metadata", {}).get("annotations", {})

        return important

    @staticmethod
    def generate_diff(
        old_resource: Dict[str, Any], new_resource: Dict[str, Any], include_full: bool = False
    ) -> Dict[str, Any]:
        if include_full:
            old_data = old_resource
            new_data = new_resource
        else:
            old_data = ResourceDiffer.extract_important_fields(old_resource)
            new_data = ResourceDiffer.extract_important_fields(new_resource)

        old_json = json.dumps(old_data, indent=2, sort_keys=True, default=str)
        new_json = json.dumps(new_data, indent=2, sort_keys=True, default=str)

        old_lines = old_json.splitlines(keepends=True)
        new_lines = new_json.splitlines(keepends=True)

        diff_lines = list(
            unified_diff(old_lines, new_lines, fromfile="previous", tofile="current", lineterm="")
        )

        changes = ResourceDiffer._parse_diff_to_changes(old_data, new_data)

        return {
            "has_changes": len(diff_lines) > 0,
            "changes": changes,
            "diff_text": "".join(diff_lines) if diff_lines else "No changes",
            "change_count": len(changes),
        }

    @staticmethod
    def _parse_diff_to_changes(
        old_data: Dict[str, Any], new_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        changes = []

        all_keys = set(old_data.keys()) | set(new_data.keys())

        for key in all_keys:
            old_value = old_data.get(key)
            new_value = new_data.get(key)

            if old_value != new_value:
                change = {"field": key, "change_type": "modified"}

                if old_value is None:
                    change["change_type"] = "added"
                    change["new_value"] = ResourceDiffer._format_value(new_value)
                elif new_value is None:
                    change["change_type"] = "removed"
                    change["old_value"] = ResourceDiffer._format_value(old_value)
                else:
                    change["old_value"] = ResourceDiffer._format_value(old_value)
                    change["new_value"] = ResourceDiffer._format_value(new_value)

                    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                        delta = new_value - old_value
                        change["delta"] = delta
                        if old_value != 0:
                            change["percent_change"] = round((delta / old_value) * 100, 2)

                changes.append(change)

        return changes

    @staticmethod
    def _format_value(value: Any, max_length: int = 100) -> str:
        if value is None:
            return "null"

        if isinstance(value, (list, dict)):
            json_str = json.dumps(value, default=str)
            if len(json_str) > max_length:
                return f"{json_str[:max_length]}... (truncated)"
            return json_str

        return str(value)

    @staticmethod
    def generate_timeline(
        versions: List[Dict[str, Any]], max_versions: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not versions:
            return []

        if max_versions:
            versions = versions[-max_versions:]

        timeline = []

        for i in range(len(versions) - 1):
            old_version = versions[i]
            new_version = versions[i + 1]

            diff = ResourceDiffer.generate_diff(old_version, new_version)

            if diff["has_changes"]:
                timeline.append(
                    {
                        "version": i + 1,
                        "timestamp": new_version.get("metadata", {}).get("creationTimestamp"),
                        "revision": new_version.get("metadata", {}).get("resourceVersion"),
                        "changes": diff["changes"],
                        "change_count": diff["change_count"],
                    }
                )

        return timeline

    @staticmethod
    def summarize_changes(changes: List[Dict[str, Any]]) -> str:
        if not changes:
            return "No changes detected"

        summary_parts = []

        added = [c for c in changes if c["change_type"] == "added"]
        removed = [c for c in changes if c["change_type"] == "removed"]
        modified = [c for c in changes if c["change_type"] == "modified"]

        if added:
            summary_parts.append(f"{len(added)} field(s) added")
        if removed:
            summary_parts.append(f"{len(removed)} field(s) removed")
        if modified:
            summary_parts.append(f"{len(modified)} field(s) modified")

        return ", ".join(summary_parts)


class DeploymentHistoryTracker:

    @staticmethod
    async def get_deployment_history(
        client,
        deployment_name: str,
        namespace: str = "default",
        max_revisions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from k8s_mcp.models import ResourceIdentifier

        deployment_id = ResourceIdentifier(
            kind="Deployment", name=deployment_name, namespace=namespace
        )

        deployment = await client.get_resource(deployment_id, use_cache=False)

        if not deployment:
            return []

        replicasets, _ = await client.list_resources(
            kind="ReplicaSet", namespace=namespace, use_cache=False
        )

        deployment_rs = []
        deployment_uid = deployment.get("metadata", {}).get("uid")

        for rs in replicasets:
            owner_refs = rs.get("metadata", {}).get("ownerReferences", [])
            for owner in owner_refs:
                if owner.get("kind") == "Deployment" and owner.get("uid") == deployment_uid:
                    revision = (
                        rs.get("metadata", {})
                        .get("annotations", {})
                        .get("deployment.kubernetes.io/revision")
                    )
                    if revision:
                        deployment_rs.append(
                            {
                                "revision": int(revision),
                                "replicaset": rs,
                                "timestamp": rs.get("metadata", {}).get("creationTimestamp"),
                                "template": rs.get("spec", {}).get("template", {}),
                            }
                        )
                    break

        deployment_rs.sort(key=lambda x: x["revision"])

        if max_revisions:
            deployment_rs = deployment_rs[-max_revisions:]

        versions = []
        for rs_data in deployment_rs:
            version = {
                "kind": "Deployment",
                "metadata": {
                    "name": deployment_name,
                    "namespace": namespace,
                    "creationTimestamp": rs_data["timestamp"],
                    "annotations": {"deployment.kubernetes.io/revision": str(rs_data["revision"])},
                },
                "spec": {
                    "template": rs_data["template"],
                    "replicas": rs_data["replicaset"].get("spec", {}).get("replicas"),
                },
            }
            versions.append(version)

        return versions

    @staticmethod
    async def get_statefulset_history(
        client,
        statefulset_name: str,
        namespace: str = "default",
        max_revisions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from k8s_mcp.models import ResourceIdentifier

        ss_id = ResourceIdentifier(kind="StatefulSet", name=statefulset_name, namespace=namespace)

        statefulset = await client.get_resource(ss_id, use_cache=False)

        if not statefulset:
            return []

        controllerrevisions, _ = await client.list_resources(
            kind="ControllerRevision", namespace=namespace, use_cache=False
        )

        ss_revisions = []
        ss_uid = statefulset.get("metadata", {}).get("uid")

        for cr in controllerrevisions:
            owner_refs = cr.get("metadata", {}).get("ownerReferences", [])
            for owner in owner_refs:
                if owner.get("kind") == "StatefulSet" and owner.get("uid") == ss_uid:
                    revision = cr.get("revision")
                    if revision:
                        ss_revisions.append(
                            {
                                "revision": revision,
                                "data": cr.get("data", {}),
                                "timestamp": cr.get("metadata", {}).get("creationTimestamp"),
                            }
                        )
                    break

        ss_revisions.sort(key=lambda x: x["revision"])

        if max_revisions:
            ss_revisions = ss_revisions[-max_revisions:]

        return ss_revisions
