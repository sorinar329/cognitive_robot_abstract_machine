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

All four cubes are registered as effect variables, not just the step's own acted-upon
cube, since the demo's actual goal is the whole stack standing, not one cube reaching its
own target height: a step whose action knocks an already-stacked cube loose while
placing its own perfectly is still a failure, and only shows up in that other cube's z.
:meth:`ActionCausalDiagnoser.diagnose` runs the diagnosis against each registered effect
in turn and keeps whichever one the observed parameters are least consistent with (see
its own docstring).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional

from krrood.adapters.json_serializer import from_json
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
)
from random_events.product_algebra import Event
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
    :meth:`ActionCausalDiagnoser.diagnose` evaluates the diagnosis once per name and
    keeps whichever one is least consistent with the observed parameters.
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
    Interventional probability the successful-attempt distribution assigns to
    :attr:`observed_value`. Zero means the value fell entirely outside what any
    successful attempt supports -- the threshold :meth:`ActionCausalDiagnoser.diagnose`
    uses to decide whether a non-primary cause is anomalous enough to also correct.
    """

    corrected_value: float
    """
    Midpoint of the region the successful-attempt distribution supports most, for
    :attr:`variable_name`.
    """

    corrected_support_probability: Optional[float] = None
    """
    Interventional probability the successful-attempt distribution assigns to
    :attr:`corrected_value`, if computed.

    Only ever populated for the primary cause: the causal circuit scores the
    recommended region's own probability just for whichever cause
    :meth:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit.diagnose_failure`
    names primary in a given call. A secondary cause's ``corrected_value`` is still a
    real recommended region (drawn from that same call's ``all_variable_results``); its
    probability at that region is just not separately computed.
    """

    def explanation(self) -> str:
        """
        :return: A one-line, human-readable account of this one correction.
        """
        correction_probability = (
            f"{self.corrected_support_probability:.4f}"
            if self.corrected_support_probability is not None
            else "not computed"
        )
        return (
            f"{self.variable_name}={self.observed_value:.4f} (support "
            f"{self.observed_support_probability:.4f}) -> {self.corrected_value:.4f} "
            f"(support {correction_probability})"
        )


@dataclass
class RootCauseDiagnosis:
    """
    One failed attempt's diagnosed causes and corrected values for them, judged against
    a single effect variable.
    """

    effect_variable_name: str
    """
    Name of the cube whose final z position the observed parameters were least
    consistent with -- see :meth:`ActionCausalDiagnoser.diagnose`. Not necessarily the
    step's own acted-upon cube: a low value here for another cube means this step's
    action is implicated in disturbing a cube it was not directly placing.
    """

    corrections: list[ParameterCorrection]
    """
    Every parameter this diagnosis corrects, primary cause first. Never empty: the
    primary cause -- the parameter least consistent with successful attempts -- is
    always included; any further parameter is included only if it is *equally*
    inconsistent (zero interventional probability, same as the primary's own threshold
    for being picked at all -- see :meth:`ActionCausalDiagnoser.diagnose`), so a failure
    with one clear cause still gets just the one correction it always did.
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


class NoRecommendationAvailable(RuntimeError):
    """
    Raised when the causal circuit could not identify a supported region for the
    primary cause variable, against any of the registered effect variables.

    Happens when every candidate cause's observed value falls outside what any
    successful attempt supports, leaving no region to recommend correcting towards.
    """

    def __init__(self, effect_names: tuple[str, ...]) -> None:
        super().__init__(
            f"no recommended region for any candidate cause, against any of "
            f"{', '.join(effect_names)}"
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
        :param config: Which parameters are candidate causes, and which are the effects.
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

    def diagnose(self, observed_parameters: dict[str, float]) -> RootCauseDiagnosis:
        """
        Identify every parameter of a failed attempt that is least consistent with
        successful attempts, and a corrected value for each.

        Runs the diagnosis once per registered effect variable (one per cube, see
        :data:`CUBE_FINAL_Z_NAMES`) and keeps the one whose primary cause has the lowest
        observed support probability -- the effect this step's action is most
        implicated in disturbing, whether or not that cube is the one it directly acted
        on.

        A single corrected parameter is often not enough: several causes can each be
        individually implausible (rather than one bad value forcing the rest), and
        ``diagnose_failure`` already scores every registered cause against the chosen
        effect in one call, not just the primary one (see its own
        ``all_variable_results``). Every other cause whose observed value also gets
        exactly zero interventional probability -- entirely outside what any successful
        attempt supports, the same unambiguous signal the primary cause itself must
        clear to be picked at all -- is corrected too, not just the single worst one.
        Causes that are merely unlikely, not impossible, are left alone: nothing marks
        them as reliably wrong, and correcting them on that weaker a signal would risk
        overwriting values that were not actually the problem.

        :param observed_parameters: The failed attempt's sampled parameter values, by
            field name. Must contain every name in
            ``self._config.cause_names``; extra keys are ignored.
        :raises NoRecommendationAvailable: If no candidate cause has a supported region
            to recommend correcting towards, against any registered effect variable.
        """
        variables_by_name = {
            variable.name: variable
            for variable in self._causal_circuit.causal_variables
        }
        observed_values = {
            variables_by_name[name]: observed_parameters[name]
            for name in self._config.cause_names
        }

        candidates: list[tuple[Variable, FailureDiagnosisResult]] = []
        for effect_variable in self._causal_circuit.effect_variables:
            diagnosis = self._causal_circuit.diagnose_failure(
                observed_values=observed_values,
                effect_variable=effect_variable,
                query_resolution=self._query_resolution,
            )
            if diagnosis.recommended_region is not None:
                candidates.append((effect_variable, diagnosis))

        if not candidates:
            raise NoRecommendationAvailable(self._config.effect_names)

        effect_variable, diagnosis = min(
            candidates, key=lambda item: item[1].interventional_probability_at_failure
        )

        corrections = [
            ParameterCorrection(
                variable_name=diagnosis.primary_cause_variable.name,
                observed_value=diagnosis.actual_value,
                observed_support_probability=diagnosis.interventional_probability_at_failure,
                corrected_value=_region_midpoint(
                    diagnosis.recommended_region, diagnosis.primary_cause_variable
                ),
                corrected_support_probability=diagnosis.interventional_probability_at_recommendation,
            )
        ]
        for variable, result in diagnosis.all_variable_results.items():
            if variable == diagnosis.primary_cause_variable:
                continue
            if (
                result["interventional_probability"] > 0
                or result["recommended_region"] is None
            ):
                continue
            corrections.append(
                ParameterCorrection(
                    variable_name=variable.name,
                    observed_value=result["actual_value"],
                    observed_support_probability=result["interventional_probability"],
                    corrected_value=_region_midpoint(
                        result["recommended_region"], variable
                    ),
                )
            )

        return RootCauseDiagnosis(
            effect_variable_name=effect_variable.name, corrections=corrections
        )

    def sample_cause_values(self) -> dict[str, float]:
        """
        Draw one sample of every candidate-cause parameter from the trained tree's own
        distribution, with every effect variable marginalized out.

        The tree is fit on successful attempts only (``training/train_jpt3.py``), so a
        value drawn this way is one the tree itself considers typical of a successful
        attempt -- unlike :class:`~pickup_place_parameterization.ParameterPrior`, which
        is a hand-specified Gaussian guess at the same thing.

        :return: One sampled value per name in ``self._config.cause_names``.
        """
        marginal_circuit = self._causal_circuit.probabilistic_circuit.marginal(
            self._causal_circuit.causal_variables
        )
        sample = marginal_circuit.sample(1)[0]
        return {
            variable.name: float(value)
            for variable, value in zip(marginal_circuit.variables, sample)
        }
