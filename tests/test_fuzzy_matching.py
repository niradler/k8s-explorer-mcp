"""Tests for fuzzy resource matching functionality."""

from k8s_explorer.fuzzy_matching import FuzzyResourceMatcher


class TestBaseNameExtraction:

    def test_extract_base_name_deployment(self):
        name = "myapp-deployment-abc123-xyz789"
        expected = "myapp-deployment"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_statefulset(self):
        name = "nginx-statefulset-0"
        expected = "nginx-statefulset"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_cronjob(self):
        name = "cronjob-1234567890-abcde"
        expected = "cronjob"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_job(self):
        name = "job-abc12"
        expected = "job"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_simple(self):
        name = "simple-pod"
        expected = "simple-pod"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_multiple_hashes(self):
        cases = [
            ("app-abc123-xyz789", "app"),
            ("service-def456-uvw012", "service"),
            ("worker-ghi789-rst345", "worker"),
        ]
        for name, expected in cases:
            assert FuzzyResourceMatcher.extract_base_name(name) == expected

    def test_extract_base_name_long_prefix(self):
        name = "my-very-long-application-name-abc123-xyz789"
        expected = "my-very-long-application-name"
        assert FuzzyResourceMatcher.extract_base_name(name) == expected


class TestSimilarityScoring:

    def test_similarity_score_identical(self):
        score = FuzzyResourceMatcher.similarity_score("hello", "hello")
        assert score == 1.0

    def test_similarity_score_similar(self):
        score = FuzzyResourceMatcher.similarity_score("hello", "hallo")
        assert 0.7 < score < 0.9

    def test_similarity_score_different(self):
        score = FuzzyResourceMatcher.similarity_score("hello", "world")
        assert score < 0.5

    def test_similarity_score_empty(self):
        score = FuzzyResourceMatcher.similarity_score("", "")
        assert score == 1.0

    def test_similarity_score_case_sensitive(self):
        score1 = FuzzyResourceMatcher.similarity_score("Test", "test")
        score2 = FuzzyResourceMatcher.similarity_score("Test", "Test")
        assert score1 < score2


class TestFindSimilarPods:

    def test_find_similar_pods_exact_base_match(self, sample_deployment_pods):
        target = "test-deployment-old999-aaa111"
        matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.7
        )

        assert len(matches) >= 3
        for match in matches:
            pod, score, reason = match
            assert score == 1.0
            assert reason == "exact_base_match"
            assert "test-deployment" in pod["metadata"]["name"]

    def test_find_similar_pods_substring_match(self, sample_deployment_pods):
        target = "test-deployment"
        matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.7
        )

        assert len(matches) >= 1
        assert matches[0][1] == 1.0
        assert matches[0][2] == "exact_base_match"

    def test_find_similar_pods_statefulset(self, sample_statefulset_pods):
        target = "database-5"
        matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_statefulset_pods, threshold=0.7
        )

        assert len(matches) == 3
        for match in matches:
            pod, score, reason = match
            assert score == 1.0
            assert reason == "exact_base_match"
            assert "database-" in pod["metadata"]["name"]

    def test_find_similar_pods_cronjob(self, sample_cronjob_pods):
        target = "backup-0000000000-zzzzz"
        matches = FuzzyResourceMatcher.find_similar_pods(target, sample_cronjob_pods, threshold=0.7)

        assert len(matches) == 2
        for match in matches:
            pod, score, reason = match
            assert score == 1.0
            assert reason == "exact_base_match"

    def test_find_similar_pods_no_match(self, sample_deployment_pods):
        target = "completely-different-pod"
        matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.9
        )

        assert len(matches) == 0

    def test_find_similar_pods_threshold_filtering(self, sample_deployment_pods):
        target = "nginx-deployment-abc123-xyz789"

        high_threshold_matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.95
        )

        low_threshold_matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.3
        )

        assert len(low_threshold_matches) >= len(high_threshold_matches)

    def test_find_similar_pods_empty_list(self):
        target = "test-pod"
        matches = FuzzyResourceMatcher.find_similar_pods(target, [], threshold=0.7)
        assert len(matches) == 0

    def test_find_similar_pods_sorting(self, sample_deployment_pods):
        target = "test-deployment-old999-aaa111"
        matches = FuzzyResourceMatcher.find_similar_pods(
            target, sample_deployment_pods, threshold=0.5
        )

        if len(matches) > 1:
            scores = [match[1] for match in matches]
            assert scores == sorted(scores, reverse=True)


class TestFindBestMatch:

    def test_find_best_match_success(self, sample_deployment_pods):
        target = "test-deployment-old999-aaa111"
        result = FuzzyResourceMatcher.find_best_match(target, sample_deployment_pods, threshold=0.7)

        assert result is not None
        pod, score, reason = result
        assert score == 1.0
        assert reason == "exact_base_match"
        assert "test-deployment" in pod["metadata"]["name"]

    def test_find_best_match_no_match(self, sample_deployment_pods):
        target = "completely-different-pod"
        result = FuzzyResourceMatcher.find_best_match(target, sample_deployment_pods, threshold=0.9)

        assert result is None

    def test_find_best_match_returns_highest_score(self):
        pods = [
            {"metadata": {"name": "nginx-abc"}},
            {"metadata": {"name": "nginx-deployment-abc123-xyz789"}},
            {"metadata": {"name": "other-service"}},
        ]

        target = "nginx-deployment-old123-abc"
        result = FuzzyResourceMatcher.find_best_match(target, pods, threshold=0.5)

        if result:
            pod, score, reason = result
            assert "nginx-deployment" in pod["metadata"]["name"]


class TestExplanations:

    def test_explain_match_exact_base(self):
        target = "nginx-deployment-old999-aaa111"
        matched = "nginx-deployment-abc123-xyz789"
        explanation = FuzzyResourceMatcher.explain_match(target, matched, 1.0, "exact_base_match")

        assert "same base name" in explanation.lower()
        assert "nginx-deployment" in explanation
        assert target in explanation
        assert matched in explanation
        assert "newer instance" in explanation.lower()

    def test_explain_match_substring(self):
        target = "nginx-deployment"
        matched = "nginx-deployment-abc123-xyz789"
        explanation = FuzzyResourceMatcher.explain_match(target, matched, 0.9, "substring_match")

        assert "overlapping name pattern" in explanation.lower()
        assert target in explanation
        assert matched in explanation

    def test_explain_match_similarity(self):
        target = "nginx-old"
        matched = "nginx-new"
        explanation = FuzzyResourceMatcher.explain_match(target, matched, 0.75, "similarity_score")

        assert "75%" in explanation
        assert target in explanation
        assert matched in explanation
        assert "similar" in explanation.lower()

    def test_explain_match_low_score(self):
        target = "app-a"
        matched = "app-b"
        explanation = FuzzyResourceMatcher.explain_match(target, matched, 0.51, "similarity_score")

        assert "51%" in explanation or "0.51" in explanation
        assert target in explanation
        assert matched in explanation


class TestEdgeCases:

    def test_pod_without_metadata(self):
        pods = [{"metadata": None}]
        target = "test"
        matches = FuzzyResourceMatcher.find_similar_pods(target, pods, threshold=0.7)
        assert len(matches) == 0

    def test_pod_without_name(self):
        pods = [{"metadata": {}}]
        target = "test"
        matches = FuzzyResourceMatcher.find_similar_pods(target, pods, threshold=0.7)
        assert len(matches) == 0

    def test_very_long_pod_names(self):
        long_name = "a" * 100 + "-abc123-xyz789"
        base = FuzzyResourceMatcher.extract_base_name(long_name)
        assert len(base) < len(long_name)

    def test_special_characters_in_name(self):
        name = "my-app_v2.0-abc123-xyz789"
        base = FuzzyResourceMatcher.extract_base_name(name)
        assert "abc123" not in base
        assert "xyz789" not in base

    def test_threshold_boundaries(self):
        pods = [{"metadata": {"name": "test-pod"}}]
        target = "test-pod"

        matches_0 = FuzzyResourceMatcher.find_similar_pods(target, pods, threshold=0.0)
        matches_1 = FuzzyResourceMatcher.find_similar_pods(target, pods, threshold=1.0)

        assert len(matches_0) >= len(matches_1)
        assert len(matches_1) > 0
