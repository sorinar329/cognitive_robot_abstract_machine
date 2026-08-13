"""
Tests for :mod:`cramera.live.model_query`, against a tiny fixture circuit of two
independent uniform variables (``dataset/model_query_circuit.json``): ``x`` on ``[0,
1]`` and ``y`` on ``[0, 2]``.
"""

from pathlib import Path

import pytest

from cramera.live.model_query import (
    EvidenceConstraint,
    EvidenceHasZeroProbability,
    ModelQueryService,
    UnknownModelVariable,
)

DATASET_DIR = Path(__file__).parent / "dataset"
MODEL_NAME = "model_query"


@pytest.fixture
def service() -> ModelQueryService:
    return ModelQueryService(models_directory=DATASET_DIR)


class TestPosteriorWithoutEvidence:
    def test_returns_the_prior_distribution(self, service):
        payload = service.posterior(MODEL_NAME, [], ["x"])["x"]
        assert payload["expectation"] == pytest.approx(0.5, abs=0.05)

    def test_answers_every_requested_query_variable(self, service):
        payload = service.posterior(MODEL_NAME, [], ["x", "y"])
        assert set(payload.keys()) == {"x", "y"}
        assert payload["y"]["expectation"] == pytest.approx(1.0, abs=0.1)


class TestPosteriorWithEvidence:
    def test_narrows_the_evidence_variables_own_posterior(self, service):
        evidence = [EvidenceConstraint(variable="x", minimum=0.2, maximum=0.4)]
        payload = service.posterior(MODEL_NAME, evidence, ["x"])["x"]
        assert payload["expectation"] == pytest.approx(0.3, abs=0.05)

    def test_zero_probability_evidence_raises(self, service):
        evidence = [EvidenceConstraint(variable="x", minimum=5.0, maximum=6.0)]
        with pytest.raises(EvidenceHasZeroProbability):
            service.posterior(MODEL_NAME, evidence, ["x"])


class TestUnknownVariables:
    def test_unknown_query_variable_raises(self, service):
        with pytest.raises(UnknownModelVariable):
            service.posterior(MODEL_NAME, [], ["not_a_real_variable"])

    def test_unknown_evidence_variable_raises(self, service):
        evidence = [
            EvidenceConstraint(variable="not_a_real_variable", minimum=0, maximum=1)
        ]
        with pytest.raises(UnknownModelVariable):
            service.posterior(MODEL_NAME, evidence, ["x"])


class TestCircuitCaching:
    def test_the_same_model_is_only_loaded_once(self, service):
        first = service._circuit(MODEL_NAME)
        second = service._circuit(MODEL_NAME)
        assert first is second


class TestEvidenceConstraintFromPayload:
    def test_builds_a_constraint_from_a_request_dict(self):
        constraint = EvidenceConstraint.from_payload(
            {"variable": "x", "minimum": 0.1, "maximum": 0.9}
        )
        assert constraint == EvidenceConstraint(variable="x", minimum=0.1, maximum=0.9)
