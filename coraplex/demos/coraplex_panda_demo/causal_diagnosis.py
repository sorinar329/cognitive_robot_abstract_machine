"""
Root-cause diagnosis of a failed stacking attempt against the joint probability trees
fitted on the successful attempts (``training/train_models.py``).

Each action's tree is wrapped in a
:class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
whose causes are the action's tunable parameters and whose effects are the final heights
of the four cubes. A failed attempt is judged against the final height of the cube it
stacked: every parameter is scored by how well the successful attempts support its
value, and the perfect value of each parameter is the one they support most.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from krrood.adapters.json_serializer import from_json
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable

# %% the diagnosed actions


class DiagnosedAction(Enum):
    """
    The actions whose failures are diagnosed, with the file of their fitted tree and the
    names of their tunable parameters among the tree's variables.
    """

    PICK_UP = (
        "pickup_jpt.json",
        (
            "pre_approach_linear_velocity",
            "grasp_linear_velocity",
            "grasp_closing_velocity",
            "grasp_stall_min_time",
            "lift_linear_velocity",
            "object_friction",
        ),
    )
    PLACE = (
        "place_jpt.json",
        (
            "transport_linear_velocity",
            "placing_linear_velocity",
            "release_opening_velocity",
            "retract_linear_velocity",
        ),
    )

    @property
    def model(self) -> Path:
        """
        The fitted tree of the action's successful parameters.
        """
        return Path(__file__).parent / "training" / "models" / self.value[0]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """
        The names of the action's tunable parameters, the causes the diagnosis scores.
        """
        return self.value[1]


class CubeHeight(Enum):
    """
    The final height of each cube, the effects among the trees' variables.
    """

    CUBE_0 = "cube0_final_z"
    CUBE_1 = "cube1_final_z"
    CUBE_2 = "cube2_final_z"
    CUBE_3 = "cube3_final_z"

    @classmethod
    def of(cls, cube_name: str) -> CubeHeight:
        """
        :param cube_name: The name of one of the cubes.
        :return: Its final height.
        """
        return cls(f"{cube_name}_final_z")


# %% diagnoses


@dataclass
class ParameterJudgement:
    """
    How well the successful attempts support one parameter's value in a failed attempt,
    and its perfect value.
    """

    parameter_name: str
    """
    Name of the parameter among the tree's variables.
    """

    observed_value: float
    """
    The failed attempt's value of the parameter.
    """

    observed_support: float
    """
    Probability the successful attempts give a window of the query resolution around
    :attr:`observed_value`; zero for a value no successful attempt used.
    """

    perfect_value: float
    """
    The middle of the region of the parameter the successful attempts support most.
    """

    perfect_support: float
    """
    Probability the successful attempts give the same window around
    :attr:`perfect_value`.
    """

    @property
    def relative_support(self) -> float:
        """
        :attr:`observed_support` as a share of :attr:`perfect_support`: 1 for a value as
        well supported as the perfect one, 0 for one no successful attempt used.

        Unlike the raw probabilities this compares across parameters: a window of fixed
        width holds less of a widely spread parameter, the friction, however typical
        its value is.
        """
        return self.observed_support / self.perfect_support

    def explanation(self) -> str:
        """
        :return: One line naming the observed value, the perfect one and their support.
        """
        return (
            f"{self.parameter_name}={self.observed_value:.4f} (support "
            f"{self.observed_support:.4f}) -> {self.perfect_value:.4f} "
            f"(support {self.perfect_support:.4f})"
        )


@dataclass
class ActionDiagnosis:
    """
    Every parameter of one action of a failed attempt, judged against the final height
    of the cube the attempt stacked.
    """

    action: DiagnosedAction
    """
    The diagnosed action.
    """

    effect_name: str
    """
    The cube height the parameters were judged against.
    """

    judgements: list[ParameterJudgement]
    """
    One judgement per parameter, least supported first.
    """

    anomaly_threshold: float
    """
    Greatest :attr:`ParameterJudgement.relative_support` of an anomalous parameter.
    """

    @property
    def anomalous(self) -> list[ParameterJudgement]:
        """
        The parameters the successful attempts support markedly less than their perfect
        value, least supported first.
        """
        return [
            judgement
            for judgement in self.judgements
            if judgement.relative_support <= self.anomaly_threshold
        ]

    @property
    def perfect_parameters(self) -> dict[str, float]:
        """
        The perfect value of every parameter of the action, by name.
        """
        return {
            judgement.parameter_name: judgement.perfect_value
            for judgement in self.judgements
        }

    def explanation(self) -> str:
        """
        :return: One line naming the anomalous parameters, primary cause first.
        """
        if not self.anomalous:
            return (
                f"no parameter anomalous against {self.effect_name}, the failure is "
                "not explained by the parameters"
            )
        primary, *others = self.anomalous
        lines = [
            f"primary cause, judged against {self.effect_name}: {primary.explanation()}"
        ]
        lines.extend(f"also anomalous: {other.explanation()}" for other in others)
        return "; ".join(lines)


# %% the diagnoser


@dataclass
class ActionCausalDiagnoser:
    """
    Diagnoses failed attempts of one action against its fitted tree.
    """

    action: DiagnosedAction
    """
    The action whose failures are diagnosed.
    """

    query_resolution: float = 0.005
    """
    Half the width of the window around a value whose probability scores it, in the
    parameter's own unit.
    """

    anomaly_threshold: float = 0.25
    """
    Greatest :attr:`ParameterJudgement.relative_support` of an anomalous parameter.
    """

    causal_circuit: CausalCircuit = field(init=False)
    """
    The action's tree, with its parameters as causes and the cube heights as effects.
    """

    def __post_init__(self) -> None:
        tree = from_json(json.loads(self.action.model.read_text()))
        circuit = tree.probabilistic_circuit
        variables_by_name = {variable.name: variable for variable in circuit.variables}
        causes = [variables_by_name[name] for name in self.action.parameter_names]
        effects = [variables_by_name[height.value] for height in CubeHeight]
        self.causal_circuit = CausalCircuit.from_probabilistic_circuit(
            circuit=circuit,
            marginal_determinism_tree=MarginalDeterminismTreeNode.from_variable_order(
                causes
            ),
            causal_variables=causes,
            effect_variables=effects,
        )
        self.causal_circuit.verify_support_determinism()

    def diagnose(
        self, observed_parameters: dict[str, float], cube_name: str
    ) -> ActionDiagnosis:
        """
        Judge every parameter of a failed attempt against the final height of the cube
        it stacked.

        :param observed_parameters: The attempt's parameter values, by name among the
            tree's variables.
        :param cube_name: The cube the attempt stacked.
        :return: The judgement of every parameter.
        """
        effect_name = CubeHeight.of(cube_name).value
        causes_by_name = {
            variable.name: variable for variable in self.causal_circuit.causal_variables
        }
        [effect] = [
            variable
            for variable in self.causal_circuit.effect_variables
            if variable.name == effect_name
        ]
        diagnosis = self.causal_circuit.diagnose_failure(
            observed_values={
                causes_by_name[name]: observed_parameters[name]
                for name in self.action.parameter_names
            },
            effect_variable=effect,
            query_resolution=self.query_resolution,
        )
        judgements = []
        for variable, result in diagnosis.all_variable_results.items():
            perfect_value = self._region_midpoint(
                result["recommended_region"], variable
            )
            judgements.append(
                ParameterJudgement(
                    parameter_name=variable.name,
                    observed_value=result["actual_value"],
                    observed_support=result["interventional_probability"],
                    perfect_value=perfect_value,
                    perfect_support=self._support(variable, effect, perfect_value),
                )
            )
        judgements.sort(key=lambda judgement: judgement.relative_support)
        return ActionDiagnosis(
            action=self.action,
            effect_name=effect_name,
            judgements=judgements,
            anomaly_threshold=self.anomaly_threshold,
        )

    def _support(self, cause: Variable, effect: Variable, value: float) -> float:
        """
        :return: The probability the successful attempts give a window of
            :attr:`query_resolution` around ``value`` of ``cause``, judged against
            ``effect``.
        """
        interventional_circuit = self.causal_circuit.backdoor_adjustment(cause, effect)
        window = SimpleEvent.from_data(
            {
                cause: closed(
                    value - self.query_resolution, value + self.query_resolution
                )
            }
        ).as_composite_set()
        return float(
            interventional_circuit.probability(
                window.fill_missing_variables_pure(interventional_circuit.variables)
            )
        )

    @staticmethod
    def _region_midpoint(region: Event, variable: Variable) -> float:
        """
        :return: The middle of ``region``'s interval of ``variable``.
        """
        interval_set = region.simple_sets[0][variable]
        [interval] = interval_set.simple_sets
        return (float(interval.lower) + float(interval.upper)) / 2.0
