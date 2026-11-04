"""
Third-generation copy of ``causal_diagnosis_v2.py``, root-cause diagnosis against the
trees ``training/train_jpt3.py`` fits from this generation's own separate collection run
(``coraplex_panda_demo_v3``, not the ``coraplex_panda_demo_v2`` archive).

Wraps the JPT trained on that run's successful attempts in the same
:class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
pattern ``causal_diagnosis_v2.py`` uses, unchanged: the causal circuit's required
``effect`` role is still played by every cube's own final z position (see
:data:`~export_successful_parameters_v3.CUBE_FINAL_Z_COLUMNS`) instead of ``step_index``,
and ``object_friction`` stays a candidate cause on the same footing as every other
tunable field -- see ``causal_diagnosis_v2.py``'s own docstring for the full reasoning.

All four cubes are registered as effect variables; :meth:`ActionCausalDiagnoser.diagnose`
judges a failure against the final z of the cube the failed step acted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from krrood.adapters.json_serializer import from_json
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable

MODEL_DIRECTORY = Path(__file__).parent / "training" / "models_v3"
"""
Where ``training/train_jpt3.py`` wrote the fitted trees.
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


CUBE_FINAL_Z_NAMES: tuple[str, ...] = (
    "cube0_final_z",
    "cube1_final_z",
    "cube2_final_z",
    "cube3_final_z",
)
"""
Names of every cube's final z position in the trained tree's variables, exactly as
``export_successful_parameters_v3.CUBE_FINAL_Z_COLUMNS`` names them.
"""


@dataclass(frozen=True)
class CausalVariableConfig:
    """
    Which of an action's parameters diagnosis searches over for the primary cause, and
    which ones play the causal circuit's required ``effect`` role.
    """

    cause_names: tuple[str, ...]
    """
    Names of the parameters diagnosis considers as candidate causes, exactly as they
    appear in the trained tree's variables.
    """

    effect_names: tuple[str, ...]
    """
    Names of the parameters that play the causal circuit's required ``effect`` role.
    """


PICKUP_CAUSAL_CONFIG = CausalVariableConfig(
    cause_names=(
        "pre_approach_linear_velocity",
        "grasp_linear_velocity",
        "grasp_closing_velocity",
        "grasp_stall_min_time",
        "lift_linear_velocity",
        "object_friction",
    ),
    effect_names=CUBE_FINAL_Z_NAMES,
)
"""
Every tunable pickup parameter, including ``object_friction``, is a candidate cause --
unchanged from :data:`~causal_diagnosis_v2.PICKUP_CAUSAL_CONFIG`'s own reasoning.
:data:`CUBE_FINAL_Z_NAMES` plays the ``effect`` role: every cube's own final z is a
directly measured outcome rather than a tunable parameter, so excluding all of them from
correction costs nothing.
"""

PLACE_CAUSAL_CONFIG = CausalVariableConfig(
    cause_names=(
        "transport_linear_velocity",
        "placing_linear_velocity",
        "release_opening_velocity",
        "retract_linear_velocity",
    ),
    effect_names=CUBE_FINAL_Z_NAMES,
)
"""
Same reasoning as :data:`PICKUP_CAUSAL_CONFIG`: every tunable place parameter stays a
candidate cause, and :data:`CUBE_FINAL_Z_NAMES` plays the ``effect`` role.
"""


@dataclass
class ParameterCorrection:
    """
    One cause parameter's diagnosed value and its recommended correction.
    """

    variable_name: str
    """
    Name of the corrected parameter.
    """

    observed_value: float
    """
    The failed attempt's actual value for :attr:`variable_name`.
    """

    observed_support_probability: float
    """
    Interventional probability the successful-attempt distribution assigns to a
    :data:`QUERY_RESOLUTION` window around :attr:`observed_value`. Zero means the value
    fell entirely outside what any successful attempt supports.
    """

    corrected_value: float
    """
    Midpoint of the region the successful-attempt distribution supports most, for
    :attr:`variable_name`.
    """

    corrected_support_probability: float
    """
    Interventional probability the successful-attempt distribution assigns to the same
    window around :attr:`corrected_value`.
    """

    @property
    def relative_support(self) -> float:
        """
        :attr:`observed_support_probability` as a share of
        :attr:`corrected_support_probability`: 1 for a value as well supported as the
        recommended one, 0 for one no successful attempt used.

        Unlike the raw probabilities, this is comparable across parameters: a window of
        fixed width holds less mass of a widely spread parameter (the friction) than of
        a narrowly spread one (a velocity), however typical the value is.
        """
        return self.observed_support_probability / self.corrected_support_probability

    def explanation(self) -> str:
        """
        :return: A one-line, human-readable account of this one correction.
        """
        return (
            f"{self.variable_name}={self.observed_value:.4f} (support "
            f"{self.observed_support_probability:.4f}) -> {self.corrected_value:.4f} "
            f"(support {self.corrected_support_probability:.4f})"
        )


@dataclass
class RootCauseDiagnosis:
    """
    One failed attempt's diagnosed causes and corrected values for them, judged against
    a single effect variable.
    """

    effect_variable_name: str
    """
    Name of the cube final z position the diagnosis was judged against.
    """

    corrections: list[ParameterCorrection]
    """
    Every parameter this diagnosis corrects, least supported first. Never empty.
    """

    @property
    def primary(self) -> ParameterCorrection:
        """
        The correction for the parameter identified as primary cause.
        """
        return self.corrections[0]

    def explanation(self) -> str:
        """
        :return: A human-readable account of every correction this diagnosis makes,
            primary cause first.
        """
        lines = [
            f"primary cause, judged against {self.effect_variable_name}: "
            f"{self.primary.explanation()}"
        ]
        lines.extend(
            f"also corrected: {correction.explanation()}"
            for correction in self.corrections[1:]
        )
        return "; ".join(lines)


class NoAnomalousParameter(RuntimeError):
    """
    Raised when every parameter of a failed attempt is about as well supported by the
    successful attempts as its recommended value, so no parameter explains the failure.
    """

    def __init__(self, effect_name: str) -> None:
        super().__init__(
            f"every parameter lies within what successful attempts used, judged "
            f"against {effect_name}"
        )


def _region_midpoint(region: Event, variable: Variable) -> float:
    """
    :param region: A recommended region, as returned on
        :attr:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.FailureDiagnosisResult.recommended_region`.
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
        anomaly_threshold: float = 0.25,
    ) -> None:
        """
        :param model_path: Where the action's trained tree is saved, as JSON.
        :param config: Which parameters are candidate causes, and which are the effects.
        :param query_resolution: See :data:`QUERY_RESOLUTION`.
        :param anomaly_threshold: Greatest :attr:`ParameterCorrection.relative_support`
            a parameter may have and still be corrected.
        :raises SupportDeterminismVerificationResult: If the loaded tree's circuit does
            not satisfy the structural property backdoor adjustment requires.
        """
        self._config = config
        self._query_resolution = query_resolution
        self._anomaly_threshold = anomaly_threshold
        self._causal_circuit = self._build_causal_circuit(model_path, config)
        self._causal_circuit.verify_support_determinism()

    @staticmethod
    def _build_causal_circuit(
        model_path: Path, config: CausalVariableConfig
    ) -> CausalCircuit:
        """
        :param model_path: Where the action's trained tree is saved, as JSON.
        :param config: Which parameters are candidate causes, and which are the effects.
        :return: A causal circuit wrapping the loaded tree, unverified.
        """
        with model_path.open() as model_file:
            tree = from_json(json.load(model_file))
        circuit = tree.probabilistic_circuit
        variables_by_name = {variable.name: variable for variable in circuit.variables}

        causal_variables = [variables_by_name[name] for name in config.cause_names]
        effect_variables = [variables_by_name[name] for name in config.effect_names]
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

    def _support_probability(
        self, variable: Variable, effect_variable: Variable, value: float
    ) -> float:
        """
        :return: The interventional probability of a :data:`QUERY_RESOLUTION` window
            around ``value`` of ``variable``, judged against ``effect_variable``.
        """
        interventional_circuit = self._causal_circuit.backdoor_adjustment(
            variable, effect_variable, [], self._query_resolution
        )
        window = SimpleEvent.from_data(
            {
                variable: closed(
                    value - self._query_resolution, value + self._query_resolution
                )
            }
        ).as_composite_set()
        return float(
            interventional_circuit.probability(
                window.fill_missing_variables_pure(interventional_circuit.variables)
            )
        )

    def diagnose(
        self, observed_parameters: dict[str, float], effect_name: str
    ) -> RootCauseDiagnosis:
        """
        Identify every parameter of a failed attempt that successful attempts support
        markedly less than its recommended value, and correct each to that value.

        A parameter is corrected when its :attr:`ParameterCorrection.relative_support`
        is at most the diagnoser's anomaly threshold; the least supported one is the
        primary cause. Correcting all of them at once matters: a retry with only the
        worst one corrected still fails on the next one.

        :param observed_parameters: The failed attempt's parameter values, by field
            name; must contain every name in the config's ``cause_names``.
        :param effect_name: The effect variable to judge against, one of the config's
            ``effect_names``.
        :raises NoAnomalousParameter: If no parameter is anomalous.
        """
        causes_by_name = {
            variable.name: variable
            for variable in self._causal_circuit.causal_variables
        }
        effect_variable = next(
            variable
            for variable in self._causal_circuit.effect_variables
            if variable.name == effect_name
        )
        diagnosis = self._causal_circuit.diagnose_failure(
            observed_values={
                causes_by_name[name]: observed_parameters[name]
                for name in self._config.cause_names
            },
            effect_variable=effect_variable,
            query_resolution=self._query_resolution,
        )

        corrections = []
        for variable, result in diagnosis.all_variable_results.items():
            if result["recommended_region"] is None:
                continue
            corrected_value = _region_midpoint(result["recommended_region"], variable)
            correction = ParameterCorrection(
                variable_name=variable.name,
                observed_value=result["actual_value"],
                observed_support_probability=result["interventional_probability"],
                corrected_value=corrected_value,
                corrected_support_probability=self._support_probability(
                    variable, effect_variable, corrected_value
                ),
            )
            if correction.relative_support <= self._anomaly_threshold:
                corrections.append(correction)

        if not corrections:
            raise NoAnomalousParameter(effect_name)
        corrections.sort(key=lambda correction: correction.relative_support)
        return RootCauseDiagnosis(
            effect_variable_name=effect_name, corrections=corrections
        )
