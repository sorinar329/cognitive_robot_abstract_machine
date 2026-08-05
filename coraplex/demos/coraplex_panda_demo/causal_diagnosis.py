"""
Root-cause diagnosis for a failed pick/place attempt.

Wraps the JPT trained on successful attempts (see ``training/train_jpt.py``) in a
:class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
Given a failed attempt's sampled parameters, it computes each parameter's interventional
support probability under the successful-attempt distribution and identifies the one
least consistent with it -- the primary cause -- together with a corrected value drawn
from the region the successful attempts actually support.

This implements the same interventional-diagnosis pattern as
``pick_place_demo_apartment_jpt_and_causal.py`` (see git history, commit ``f1adf95ee``),
adapted to this demo's simpler action parameters. That demo also had a directly measured
outcome (the placed object's final position) to use as the causal circuit's ``effect``
variable; this demo's training data has no such outcome column, only the sampled inputs
themselves (see ``export_successful_parameters.py``). Each action's config below instead
designates one of its own parameters as the ``effect`` -- see
:data:`PICKUP_CAUSAL_CONFIG`/:data:`PLACE_CAUSAL_CONFIG` for which, and why. The circuit
does not condition on that variable's value; it only needs some variable to play the
role structurally. What actually drives diagnosis is each cause variable's own
interventional support probability, not the effect variable's value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from krrood.adapters.json_serializer import from_json
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
)
from random_events.product_algebra import Event
from random_events.variable import Variable

MODEL_DIRECTORY = Path(__file__).parent / "training" / "models"
"""
Where ``training/train_jpt.py`` wrote the fitted trees.
"""

PICKUP_MODEL_PATH = MODEL_DIRECTORY / "pickup_jpt.json"
"""
The trained :class:`PickUpAction` parameter tree.
"""

PLACE_MODEL_PATH = MODEL_DIRECTORY / "place_jpt.json"
"""
The trained :class:`PlaceAction` parameter tree.
"""

QUERY_RESOLUTION = 0.005
"""
Width (in the variable's own units) of the interval each interventional query is
evaluated over, centered on the queried value -- see
:meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.diagnose_failure`.
"""


@dataclass(frozen=True)
class CausalVariableConfig:
    """
    Which of an action's parameters diagnosis searches over for the primary cause, and
    which one plays the causal circuit's required ``effect`` role.
    """

    cause_names: tuple[str, ...]
    """
    Names of the parameters diagnosis considers as candidate causes, exactly as they
    appear in the trained tree's variables.
    """

    effect_name: str
    """
    Name of the parameter that plays the causal circuit's required ``effect`` role.
    """


PICKUP_CAUSAL_CONFIG = CausalVariableConfig(
    cause_names=(
        "pre_approach_linear_velocity",
        "grasp_linear_velocity",
        "grasp_closing_velocity",
        "grasp_stall_min_time",
        "lift_linear_velocity",
    ),
    effect_name="object_friction",
)
"""
The velocity/timing parameters are what a caller chooses per attempt, so they are the
candidate causes. ``object_friction`` is a property of the object being picked rather
than a choice the caller makes, so it plays the ``effect`` role instead.
"""

PLACE_CAUSAL_CONFIG = CausalVariableConfig(
    cause_names=(
        "transport_linear_velocity",
        "placing_linear_velocity",
        "release_opening_velocity",
    ),
    effect_name="retract_linear_velocity",
)
"""
``retract_linear_velocity`` is the last of :class:`PlaceAction`'s tunable speeds to take
effect, so it plays the ``effect`` role while the three earlier ones are the candidate
causes.
"""


@dataclass
class RootCauseDiagnosis:
    """
    One failed attempt's primary cause and a corrected value for it.
    """

    variable_name: str
    """
    Name of the parameter identified as the primary cause.
    """

    observed_value: float
    """
    The failed attempt's actual value for :attr:`variable_name`.
    """

    observed_support_probability: float
    """
    Interventional probability the successful-attempt distribution assigns to
    :attr:`observed_value`. Zero means the value fell entirely outside what any
    successful attempt supports.
    """

    corrected_value: float
    """
    Midpoint of the region the successful-attempt distribution supports most, for
    :attr:`variable_name`.
    """

    corrected_support_probability: float
    """
    Interventional probability the successful-attempt distribution assigns to
    :attr:`corrected_value`.
    """

    def explanation(self) -> str:
        """
        :return: A one-line, human-readable account of the diagnosis.
        """
        return (
            f"{self.variable_name}={self.observed_value:.4f} is the parameter least "
            f"consistent with successful attempts (support probability "
            f"{self.observed_support_probability:.4f}); successful attempts instead "
            f"support values near {self.corrected_value:.4f} (support probability "
            f"{self.corrected_support_probability:.4f})"
        )


class NoRecommendationAvailable(RuntimeError):
    """
    Raised when the causal circuit could not identify a supported region for the
    primary cause variable.

    Happens when every candidate cause's observed value falls outside what any
    successful attempt supports, leaving no region to recommend correcting towards.
    """

    def __init__(self, diagnosis: FailureDiagnosisResult) -> None:
        super().__init__(
            f"no recommended region for primary cause "
            f"{diagnosis.primary_cause_variable.name}"
        )


def _region_midpoint(region: Event, variable: Variable) -> float:
    """
    :param region: A recommended region, as returned on
        :attr:`~FailureDiagnosisResult.recommended_region`.
    :param variable: The variable to read the region's interval for.
    :return: The midpoint of ``region``'s interval for ``variable``.
    """
    simple_set = region.simple_sets[0]
    interval_set = simple_set[variable]
    interval = (
        interval_set.simple_sets[0]
        if hasattr(interval_set, "simple_sets")
        else interval_set
    )
    return (float(interval.lower) + float(interval.upper)) / 2.0


class ActionCausalDiagnoser:
    """
    Diagnoses failed attempts of one action against its trained parameter tree.

    Builds its :class:`CausalCircuit` once, at construction time, and reuses it for
    every :meth:`diagnose` call.
    """

    def __init__(
        self,
        model_path: Path,
        config: CausalVariableConfig,
        query_resolution: float = QUERY_RESOLUTION,
    ) -> None:
        """
        :param model_path: Where the action's trained tree is saved, as JSON.
        :param config: Which parameters are candidate causes, and which is the effect.
        :param query_resolution: See :data:`QUERY_RESOLUTION`.
        :raises SupportDeterminismVerificationResult: If the loaded tree's circuit does
            not satisfy the structural property backdoor adjustment requires.
        """
        self._config = config
        self._query_resolution = query_resolution
        self._causal_circuit = self._build_causal_circuit(model_path, config)
        self._causal_circuit.verify_support_determinism()

    @staticmethod
    def _build_causal_circuit(
        model_path: Path, config: CausalVariableConfig
    ) -> CausalCircuit:
        """
        :param model_path: Where the action's trained tree is saved, as JSON.
        :param config: Which parameters are candidate causes, and which is the effect.
        :return: A causal circuit wrapping the loaded tree, unverified.
        """
        with model_path.open() as model_file:
            tree = from_json(json.load(model_file))
        circuit = tree.probabilistic_circuit
        variables_by_name = {variable.name: variable for variable in circuit.variables}

        causal_variables = [variables_by_name[name] for name in config.cause_names]
        effect_variables = [variables_by_name[config.effect_name]]
        determinism_tree = MarginalDeterminismTreeNode.from_causal_graph(
            causal_variables=causal_variables,
            effect_variables=effect_variables,
            causal_priority_order=causal_variables,
        )
        return CausalCircuit.from_probabilistic_circuit(
            circuit=circuit,
            marginal_determinism_tree=determinism_tree,
            causal_variables=causal_variables,
            effect_variables=effect_variables,
        )

    def diagnose(self, observed_parameters: dict[str, float]) -> RootCauseDiagnosis:
        """
        Identify which parameter of a failed attempt is least consistent with
        successful attempts, and a corrected value for it.

        :param observed_parameters: The failed attempt's sampled parameter values, by
            field name. Must contain every name in
            ``self._config.cause_names``; extra keys are ignored.
        :raises NoRecommendationAvailable: If no candidate cause has a supported region
            to recommend correcting towards.
        """
        variables_by_name = {
            variable.name: variable
            for variable in self._causal_circuit.causal_variables
        }
        observed_values = {
            variables_by_name[name]: observed_parameters[name]
            for name in self._config.cause_names
        }
        diagnosis = self._causal_circuit.diagnose_failure(
            observed_values=observed_values,
            effect_variable=self._causal_circuit.effect_variables[0],
            query_resolution=self._query_resolution,
        )
        if diagnosis.recommended_region is None:
            raise NoRecommendationAvailable(diagnosis)

        corrected_value = _region_midpoint(
            diagnosis.recommended_region, diagnosis.primary_cause_variable
        )
        return RootCauseDiagnosis(
            variable_name=diagnosis.primary_cause_variable.name,
            observed_value=diagnosis.actual_value,
            observed_support_probability=diagnosis.interventional_probability_at_failure,
            corrected_value=corrected_value,
            corrected_support_probability=diagnosis.interventional_probability_at_recommendation,
        )
