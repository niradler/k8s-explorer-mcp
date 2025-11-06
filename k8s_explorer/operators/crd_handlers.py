"""CRD and Operator-specific relationship handlers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..models import RelationshipType, ResourceIdentifier
from ..relationships import ResourceRelationship

logger = logging.getLogger(__name__)


class OperatorHandler(ABC):
    """Base class for operator-specific handlers."""

    @abstractmethod
    def can_handle(self, resource: Dict[str, Any]) -> bool:
        """Check if this handler can process the resource."""
        pass

    @abstractmethod
    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover operator-specific relationships."""
        pass


class ArgoWorkflowsHandler(OperatorHandler):
    """Handler for Argo Workflows CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        kind = resource.get("kind", "")
        return "argoproj.io" in api_version and kind in [
            "Workflow",
            "CronWorkflow",
            "WorkflowTemplate",
        ]

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Argo Workflow relationships."""
        relationships = []
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})
        resource_id = ResourceIdentifier(
            kind=kind,
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        labels = metadata.get("labels", {})
        workflow_name = labels.get("workflows.argoproj.io/workflow")

        if workflow_name and kind in ["Pod"]:
            workflow_id = ResourceIdentifier(
                kind="Workflow",
                name=workflow_name,
                namespace=metadata.get("namespace"),
                api_version="argoproj.io/v1alpha1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=workflow_id,
                    relationship_type=RelationshipType.CRD,
                    details="Created by Argo Workflow",
                )
            )

        spec = resource.get("spec", {})
        if kind == "Workflow" and "templates" in spec:
            for template in spec.get("templates", []):
                if "resource" in template:
                    resource_manifest = template["resource"].get("manifest", "")
                    logger.debug(f"Workflow creates resources: {resource_manifest[:100]}")

        return relationships


class ArgoCDHandler(OperatorHandler):
    """Handler for Argo CD CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        kind = resource.get("kind", "")
        annotations = resource.get("metadata", {}).get("annotations", {})

        return ("argoproj.io" in api_version and kind == "Application") or any(
            k.startswith("argocd.argoproj.io") for k in annotations.keys()
        )

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Argo CD relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        annotations = metadata.get("annotations", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        app_instance = annotations.get("argocd.argoproj.io/instance")
        if app_instance:
            app_id = ResourceIdentifier(
                kind="Application",
                name=app_instance,
                namespace="argocd",
                api_version="argoproj.io/v1alpha1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=app_id,
                    relationship_type=RelationshipType.CRD,
                    details="Managed by ArgoCD Application",
                )
            )

        return relationships


class AirflowHandler(OperatorHandler):
    """Handler for Apache Airflow Kubernetes resources."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        labels = resource.get("metadata", {}).get("labels", {})
        return (
            "airflow" in labels.get("app", "").lower() or "dag_id" in labels or "task_id" in labels
        )

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Airflow-related relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
        )

        dag_id = labels.get("dag_id")
        task_id = labels.get("task_id")

        if dag_id:
            label_selector = f"dag_id={dag_id}"
            try:
                related_pods = await client.list_resources(
                    kind="Pod",
                    namespace=metadata.get("namespace"),
                    label_selector=label_selector,
                    use_cache=True,
                )

                for pod in related_pods[:10]:
                    pod_metadata = pod.get("metadata", {})
                    if pod_metadata.get("name") != metadata.get("name"):
                        pod_id = ResourceIdentifier(
                            kind="Pod",
                            name=pod_metadata.get("name"),
                            namespace=pod_metadata.get("namespace"),
                        )
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=pod_id,
                                relationship_type=RelationshipType.CRD,
                                details=f"Same Airflow DAG: {dag_id}, Task: {task_id or 'N/A'}",
                            )
                        )
            except Exception as e:
                logger.debug(f"Error discovering Airflow relationships: {e}")

        return relationships


class KnativeHandler(OperatorHandler):
    """Handler for Knative CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        return "knative.dev" in api_version or "serving.knative.dev" in api_version

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Knative relationships."""
        relationships = []
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})

        resource_id = ResourceIdentifier(
            kind=kind,
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        if kind == "Service":
            labels = metadata.get("labels", {})
            serving_service = labels.get("serving.knative.dev/service")

            if serving_service:
                try:
                    label_selector = f"serving.knative.dev/service={serving_service}"
                    pods = await client.list_resources(
                        kind="Pod",
                        namespace=metadata.get("namespace"),
                        label_selector=label_selector,
                        use_cache=True,
                    )

                    for pod in pods[:10]:
                        pod_id = ResourceIdentifier(
                            kind="Pod",
                            name=pod.get("metadata", {}).get("name"),
                            namespace=pod.get("metadata", {}).get("namespace"),
                        )
                        relationships.append(
                            ResourceRelationship(
                                source=resource_id,
                                target=pod_id,
                                relationship_type=RelationshipType.CRD,
                                details="Knative Serving pods",
                            )
                        )
                except Exception as e:
                    logger.debug(f"Error discovering Knative relationships: {e}")

        return relationships


class FluxCDHandler(OperatorHandler):
    """Handler for FluxCD CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        return (
            "fluxcd.io" in api_version
            or "kustomize.toolkit.fluxcd.io" in api_version
            or "helm.toolkit.fluxcd.io" in api_version
        )

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover FluxCD relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        annotations = metadata.get("annotations", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        kustomization = labels.get("kustomize.toolkit.fluxcd.io/name")
        helm_release = labels.get("helm.toolkit.fluxcd.io/name")

        if kustomization:
            flux_id = ResourceIdentifier(
                kind="Kustomization",
                name=kustomization,
                namespace=metadata.get("namespace"),
                api_version="kustomize.toolkit.fluxcd.io/v1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=flux_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Managed by FluxCD Kustomization: {kustomization}",
                )
            )

        if helm_release:
            flux_id = ResourceIdentifier(
                kind="HelmRelease",
                name=helm_release,
                namespace=metadata.get("namespace"),
                api_version="helm.toolkit.fluxcd.io/v2beta1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=flux_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Managed by FluxCD HelmRelease: {helm_release}",
                )
            )

        return relationships


class IstioHandler(OperatorHandler):
    """Handler for Istio CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        labels = resource.get("metadata", {}).get("labels", {})
        return "istio.io" in api_version or "service.istio.io" in labels.get("", "")

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Istio relationships."""
        relationships = []
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})

        resource_id = ResourceIdentifier(
            kind=kind,
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        if kind == "VirtualService":
            spec = resource.get("spec", {})
            for host in spec.get("hosts", []):
                logger.debug(f"VirtualService routes to host: {host}")

        labels = metadata.get("labels", {})
        if "app" in labels and kind == "Pod":
            try:
                vs_list = await client.list_resources(
                    kind="VirtualService", namespace=metadata.get("namespace"), use_cache=True
                )
                for vs in vs_list[:5]:
                    vs_metadata = vs.get("metadata", {})
                    vs_id = ResourceIdentifier(
                        kind="VirtualService",
                        name=vs_metadata.get("name"),
                        namespace=vs_metadata.get("namespace"),
                        api_version="networking.istio.io/v1beta1",
                    )
                    relationships.append(
                        ResourceRelationship(
                            source=resource_id,
                            target=vs_id,
                            relationship_type=RelationshipType.CRD,
                            details="Istio traffic routing",
                        )
                    )
            except Exception as e:
                logger.debug(f"Error discovering Istio relationships: {e}")

        return relationships


class CertManagerHandler(OperatorHandler):
    """Handler for cert-manager CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        annotations = resource.get("metadata", {}).get("annotations", {})
        return "cert-manager.io" in api_version or "cert-manager.io/certificate-name" in annotations

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover cert-manager relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        annotations = metadata.get("annotations", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        cert_name = annotations.get("cert-manager.io/certificate-name")
        if cert_name:
            cert_id = ResourceIdentifier(
                kind="Certificate",
                name=cert_name,
                namespace=metadata.get("namespace"),
                api_version="cert-manager.io/v1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=cert_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Uses cert-manager Certificate: {cert_name}",
                )
            )

        return relationships


class TektonHandler(OperatorHandler):
    """Handler for Tekton Pipelines CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        labels = resource.get("metadata", {}).get("labels", {})
        return "tekton.dev" in api_version or "tekton.dev/pipeline" in labels

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Tekton relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        pipeline = labels.get("tekton.dev/pipeline")
        pipeline_run = labels.get("tekton.dev/pipelineRun")
        task_run = labels.get("tekton.dev/taskRun")

        if pipeline:
            pipeline_id = ResourceIdentifier(
                kind="Pipeline",
                name=pipeline,
                namespace=metadata.get("namespace"),
                api_version="tekton.dev/v1beta1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=pipeline_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Part of Tekton Pipeline: {pipeline}",
                )
            )

        if pipeline_run:
            pr_id = ResourceIdentifier(
                kind="PipelineRun",
                name=pipeline_run,
                namespace=metadata.get("namespace"),
                api_version="tekton.dev/v1beta1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=pr_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Created by PipelineRun: {pipeline_run}",
                )
            )

        return relationships


class SparkOperatorHandler(OperatorHandler):
    """Handler for Spark Operator CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        labels = resource.get("metadata", {}).get("labels", {})
        return "sparkoperator.k8s.io" in api_version or "sparkoperator.k8s.io/app-name" in labels

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Spark Operator relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        spark_app = labels.get("sparkoperator.k8s.io/app-name")
        spark_role = labels.get("spark-role")

        if spark_app:
            spark_id = ResourceIdentifier(
                kind="SparkApplication",
                name=spark_app,
                namespace=metadata.get("namespace"),
                api_version="sparkoperator.k8s.io/v1beta2",
            )
            role_info = f", Role: {spark_role}" if spark_role else ""
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=spark_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Spark Application: {spark_app}{role_info}",
                )
            )

        return relationships


class KedaHandler(OperatorHandler):
    """Handler for KEDA (autoscaling) CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        annotations = resource.get("metadata", {}).get("annotations", {})
        return "keda.sh" in api_version or any("keda.sh" in k for k in annotations.keys())

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover KEDA relationships."""
        relationships = []
        kind = resource.get("kind")
        metadata = resource.get("metadata", {})

        resource_id = ResourceIdentifier(
            kind=kind,
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        if kind == "ScaledObject":
            spec = resource.get("spec", {})
            scale_target = spec.get("scaleTargetRef", {})
            target_name = scale_target.get("name")
            target_kind = scale_target.get("kind", "Deployment")

            if target_name:
                target_id = ResourceIdentifier(
                    kind=target_kind, name=target_name, namespace=metadata.get("namespace")
                )
                relationships.append(
                    ResourceRelationship(
                        source=resource_id,
                        target=target_id,
                        relationship_type=RelationshipType.CRD,
                        details=f"KEDA autoscaling for {target_kind}",
                    )
                )

        return relationships


class VeleroHandler(OperatorHandler):
    """Handler for Velero (backup) CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        labels = resource.get("metadata", {}).get("labels", {})
        return "velero.io" in api_version or "velero.io/backup-name" in labels

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Velero relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        backup_name = labels.get("velero.io/backup-name")
        restore_name = labels.get("velero.io/restore-name")

        if backup_name:
            backup_id = ResourceIdentifier(
                kind="Backup", name=backup_name, namespace="velero", api_version="velero.io/v1"
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=backup_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Included in Velero Backup: {backup_name}",
                )
            )

        if restore_name:
            restore_id = ResourceIdentifier(
                kind="Restore", name=restore_name, namespace="velero", api_version="velero.io/v1"
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=restore_id,
                    relationship_type=RelationshipType.CRD,
                    details=f"Restored by Velero Restore: {restore_name}",
                )
            )

        return relationships


class PrometheusOperatorHandler(OperatorHandler):
    """Handler for Prometheus Operator CRDs."""

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        annotations = resource.get("metadata", {}).get("annotations", {})
        return "monitoring.coreos.com" in api_version or "prometheus.io/scrape" in annotations

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Discover Prometheus Operator relationships."""
        relationships = []
        metadata = resource.get("metadata", {})
        annotations = metadata.get("annotations", {})
        labels = metadata.get("labels", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        if annotations.get("prometheus.io/scrape") == "true":
            prometheus_id = ResourceIdentifier(
                kind="ServiceMonitor",
                name=f"{metadata.get('name')}-monitor",
                namespace=metadata.get("namespace"),
                api_version="monitoring.coreos.com/v1",
            )
            relationships.append(
                ResourceRelationship(
                    source=resource_id,
                    target=prometheus_id,
                    relationship_type=RelationshipType.CRD,
                    details="Monitored by Prometheus",
                )
            )

        return relationships


class GenericCRDHandler(OperatorHandler):
    """
    AI-powered fallback handler for unknown CRDs.
    Uses LLM sampling to intelligently discover relationships.
    """

    def __init__(self, llm_sampler=None):
        """Initialize with optional LLM sampler."""
        self.llm_sampler = llm_sampler
        self._label_patterns = [
            "app.kubernetes.io/instance",
            "app.kubernetes.io/name",
            "app.kubernetes.io/component",
            "app.kubernetes.io/part-of",
            "app.kubernetes.io/managed-by",
        ]

    def can_handle(self, resource: Dict[str, Any]) -> bool:
        api_version = resource.get("apiVersion", "")
        return "/" in api_version and api_version.split("/")[0] not in [
            "v1",
            "apps",
            "batch",
            "networking.k8s.io",
        ]

    async def discover_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """
        Discover relationships using:
        1. Common label patterns
        2. Owner references
        3. LLM-powered analysis (if available)
        """
        relationships = []
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {})
        annotations = metadata.get("annotations", {})

        resource_id = ResourceIdentifier(
            kind=resource.get("kind"),
            name=metadata.get("name"),
            namespace=metadata.get("namespace"),
            api_version=resource.get("apiVersion"),
        )

        for pattern in self._label_patterns:
            if pattern in labels and labels[pattern]:
                value = labels[pattern]
                try:
                    label_selector = f"{pattern}={value}"
                    related = await client.list_resources(
                        kind="Pod",
                        namespace=metadata.get("namespace"),
                        label_selector=label_selector,
                        use_cache=True,
                    )

                    if related:
                        for res in related[:5]:
                            res_metadata = res.get("metadata", {})
                            if res_metadata.get("name") != metadata.get("name"):
                                rel_id = ResourceIdentifier(
                                    kind="Pod",
                                    name=res_metadata.get("name"),
                                    namespace=res_metadata.get("namespace"),
                                )
                                relationships.append(
                                    ResourceRelationship(
                                        source=resource_id,
                                        target=rel_id,
                                        relationship_type=RelationshipType.CRD,
                                        details=f"Related by label {pattern}={value}",
                                    )
                                )
                except Exception as e:
                    logger.debug(f"Error discovering generic CRD relationships: {e}")

        if self.llm_sampler:
            try:
                ai_rels = await self._discover_with_llm(resource, client)
                relationships.extend(ai_rels)
            except Exception as e:
                logger.debug(f"LLM-powered discovery failed: {e}")

        return relationships

    async def _discover_with_llm(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """Use LLM to discover non-obvious relationships."""
        if not self.llm_sampler:
            return []

        metadata = resource.get("metadata", {})
        spec = resource.get("spec", {})

        prompt = f"""Analyze this Kubernetes CRD resource and identify potential relationships:

Kind: {resource.get('kind')}
ApiVersion: {resource.get('apiVersion')}
Labels: {metadata.get('labels', {})}
Annotations: {metadata.get('annotations', {})}
Spec keys: {list(spec.keys()) if spec else []}

What other Kubernetes resources might this CRD manage or relate to?
Respond with a JSON list of relationship hints:
[
  {{"kind": "Pod", "label_selector": "app=example", "reason": "manages pods"}},
  {{"kind": "Service", "name_pattern": "example-svc", "reason": "creates service"}}
]
"""

        try:
            response = await self.llm_sampler.sample(prompt)
            logger.info(f"LLM discovered relationships for {resource.get('kind')}: {response}")
        except Exception as e:
            logger.debug(f"LLM sampling error: {e}")

        return []


class CRDOperatorRegistry:
    """
    Registry for CRD operator handlers.

    Supports 11 built-in operators + AI-powered fallback for unknown CRDs:
    1. Helm
    2. ArgoCD
    3. Argo Workflows
    4. Apache Airflow
    5. Knative
    6. FluxCD
    7. Istio
    8. cert-manager
    9. Tekton
    10. Spark Operator
    11. KEDA
    12. Velero
    13. Prometheus Operator
    + Generic (AI-powered)
    """

    def __init__(self, llm_sampler=None):
        """
        Initialize registry with built-in handlers.

        Args:
            llm_sampler: Optional LLM sampler for generic CRD discovery
        """
        self.handlers: List[OperatorHandler] = [
            ArgoWorkflowsHandler(),
            ArgoCDHandler(),
            AirflowHandler(),
            KnativeHandler(),
            FluxCDHandler(),
            IstioHandler(),
            CertManagerHandler(),
            TektonHandler(),
            SparkOperatorHandler(),
            KedaHandler(),
            VeleroHandler(),
            PrometheusOperatorHandler(),
            GenericCRDHandler(llm_sampler),
        ]
        logger.info(f"Initialized CRD registry with {len(self.handlers)} handlers")

    def register_handler(self, handler: OperatorHandler):
        """Register a custom handler (will be checked before generic handler)."""
        self.handlers.insert(-1, handler)
        logger.info(f"Registered custom handler: {handler.__class__.__name__}")

    async def discover_crd_relationships(
        self, resource: Dict[str, Any], client
    ) -> List[ResourceRelationship]:
        """
        Discover relationships using appropriate handlers.

        Tries handlers in order until one matches. The GenericCRDHandler
        (with AI fallback) is always tried last.
        """
        relationships = []
        handled = False

        for handler in self.handlers:
            if handler.can_handle(resource):
                try:
                    rels = await handler.discover_relationships(resource, client)
                    relationships.extend(rels)
                    handled = True
                    if not isinstance(handler, GenericCRDHandler):
                        logger.debug(f"{handler.__class__.__name__} handled {resource.get('kind')}")
                except Exception as e:
                    logger.warning(f"Handler {handler.__class__.__name__} failed: {e}")

        if not handled:
            logger.debug(
                f"No specific handler for {resource.get('kind')}/{resource.get('apiVersion')}"
            )

        return relationships

    def get_supported_operators(self) -> List[str]:
        """Get list of supported operators."""
        return [
            "Helm",
            "ArgoCD",
            "Argo Workflows",
            "Apache Airflow",
            "Knative",
            "FluxCD",
            "Istio",
            "cert-manager",
            "Tekton Pipelines",
            "Spark Operator",
            "KEDA",
            "Velero",
            "Prometheus Operator",
            "Generic (AI-powered fallback for any CRD)",
        ]
