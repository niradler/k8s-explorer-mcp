"""Smart resource matching for cases when exact names are not available."""

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FuzzyResourceMatcher:

    DEPLOYMENT_SUFFIX_PATTERN = re.compile(r"-[a-z0-9]{5,10}-[a-z0-9]{5,6}$")
    STATEFULSET_SUFFIX_PATTERN = re.compile(r"-\d+$")
    JOB_SUFFIX_PATTERN = re.compile(r"-[a-z0-9]{5,6}$")
    CRONJOB_SUFFIX_PATTERN = re.compile(r"-\d{8,10}-[a-z0-9]{5,6}$")

    @staticmethod
    def extract_base_name(pod_name: str) -> str:
        base = pod_name

        if FuzzyResourceMatcher.CRONJOB_SUFFIX_PATTERN.search(pod_name):
            base = FuzzyResourceMatcher.CRONJOB_SUFFIX_PATTERN.sub("", pod_name)
        elif FuzzyResourceMatcher.DEPLOYMENT_SUFFIX_PATTERN.search(pod_name):
            base = FuzzyResourceMatcher.DEPLOYMENT_SUFFIX_PATTERN.sub("", pod_name)
        elif FuzzyResourceMatcher.STATEFULSET_SUFFIX_PATTERN.search(pod_name):
            base = FuzzyResourceMatcher.STATEFULSET_SUFFIX_PATTERN.sub("", pod_name)
        elif FuzzyResourceMatcher.JOB_SUFFIX_PATTERN.search(pod_name):
            base = FuzzyResourceMatcher.JOB_SUFFIX_PATTERN.sub("", pod_name)

        return base

    @staticmethod
    def similarity_score(str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1, str2).ratio()

    @staticmethod
    def find_similar_pods(
        target_name: str, available_pods: List[Dict[str, Any]], threshold: float = 0.7
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        base_name = FuzzyResourceMatcher.extract_base_name(target_name)

        matches = []
        for pod in available_pods:
            metadata = pod.get("metadata")
            if metadata is None or not isinstance(metadata, dict):
                continue

            pod_name = metadata.get("name", "")
            if not pod_name:
                continue

            pod_base = FuzzyResourceMatcher.extract_base_name(pod_name)

            if base_name == pod_base:
                score = 1.0
                reason = "exact_base_match"
            elif base_name in pod_name or pod_base in target_name:
                score = 0.9
                reason = "substring_match"
            else:
                score = FuzzyResourceMatcher.similarity_score(base_name, pod_base)
                reason = "similarity_score"

            if score >= threshold:
                matches.append((pod, score, reason))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    @staticmethod
    def find_best_match(
        target_name: str, available_pods: List[Dict[str, Any]], threshold: float = 0.7
    ) -> Optional[Tuple[Dict[str, Any], float, str]]:
        matches = FuzzyResourceMatcher.find_similar_pods(target_name, available_pods, threshold)

        if matches:
            return matches[0]
        return None

    @staticmethod
    def explain_match(target_name: str, matched_name: str, score: float, reason: str) -> str:
        if reason == "exact_base_match":
            base = FuzzyResourceMatcher.extract_base_name(target_name)
            return (
                f"Pod '{target_name}' not found, but found '{matched_name}' "
                f"with same base name '{base}'. "
                f"This is likely a newer instance of the same pod."
            )
        elif reason == "substring_match":
            return (
                f"Pod '{target_name}' not found, but found similar pod '{matched_name}' "
                f"with overlapping name pattern."
            )
        else:
            return (
                f"Pod '{target_name}' not found, but found similar pod '{matched_name}' "
                f"(similarity: {score:.0%}). This might be a renamed or recreated version."
            )
