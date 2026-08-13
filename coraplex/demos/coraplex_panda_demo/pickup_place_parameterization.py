"""
Randomized sampling of PickUpAction/PlaceAction velocity/timing parameters, so repeated
stacking attempts in ``demo2.py`` produce varied values instead of always using the
fixed dataclass defaults -- a prerequisite for later training a probabilistic model on
which parameter values led to a successful stack.

Reuses the workspace's existing "underspecified parameters" framework
(``krrood.parametrization``) rather than a bespoke random sampler, following the same
pattern already used in
``coraplex.training_environments.training_environment.MoveToReachTrainingEnvironment``.
No trained model is used yet -- every sample is drawn from a hand-specified Gaussian
prior, truncated to a fixed range. Fitting a real model on collected data is a follow-up
step once enough varied data exists.

Each prior centers on a value that stacks reliably but is spread wide enough for its
tails to reach the values observed to break a grasp or knock the stack over, so runs
produce both outcomes. Priors held entirely inside the reliable band produced a
near-100% success rate, and ones centered on the aggressive end produced none; neither
says where the boundary lies, which a model can only learn from data containing both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from random_events.variable import Continuous, Variable
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(frozen=True)
class ParameterPrior:
    """
    A single float parameter's sampling prior: a Gaussian centered on ``mean`` with
    spread ``std``, truncated to ``[low, high]`` (the truncation is what actually bounds
    the samples -- the Gaussian shape only biases how densely the range is covered,
    concentrating samples around ``mean`` rather than spreading them uniformly).
    """

    mean: float
    std: float
    low: float
    high: float


PICKUP_PARAMETER_PRIORS: Dict[str, ParameterPrior] = {
    "pre_approach_linear_velocity": ParameterPrior(
        mean=0.115, std=0.03, low=0.06, high=0.24
    ),
    # 0.2 was observed to cause a large InfeasibleException failure spike
    # (~1/10 success). Picks stayed reliable around 0.05-0.085, so the bulk sits
    # there and only the far tail approaches that spike.
    "final_approach_linear_velocity": ParameterPrior(
        mean=0.045, std=0.035, low=0.01, high=0.22
    ),
    # The finger joints' physical velocity limit is 0.2 m/s, past which the
    # fingers slam shut fast enough to punt the cube out of the grasp instead of
    # closing on it.
    "grasp_closing_velocity": ParameterPrior(mean=0.1, std=0.035, low=0.03, high=0.22),
    # Lifting at 0.12 kept the stack standing while 0.18 did not, so the upper
    # tail crosses that gap.
    "lift_linear_velocity": ParameterPrior(mean=0.125, std=0.035, low=0.05, high=0.28),
    # 0.3 is the validated floor for stall detection; below it the grasp is
    # called complete before the fingers have settled on the cube.
    "grasp_stall_minimum_time": ParameterPrior(
        mean=0.4, std=0.1, low=0.15, high=0.85
    ),
    # 1.5 (sliding friction) is the cube geoms' own MJCF default. Below roughly
    # 0.3 the cube slides straight out of the fingers, which the lower tail
    # reaches, while staying above 0 and below values MuJoCo's pyramidal
    # friction cone starts struggling with numerically.
    "object_friction": ParameterPrior(mean=1.2, std=0.5, low=0.15, high=2.5),
}
"""
Sampling prior for each of :class:`PickUpAction`'s tunable velocity/timing fields.
"""

PLACE_PARAMETER_PRIORS: Dict[str, ParameterPrior] = {
    # Transporting at 0.078 left the stack standing while 0.12 did not.
    "transport_linear_velocity": ParameterPrior(
        mean=0.08, std=0.028, low=0.03, high=0.18
    ),
    # High values drive the cube down onto the stack hard enough to scatter what
    # is already standing: 0.05 held, 0.08 did not.
    "placing_linear_velocity": ParameterPrior(
        mean=0.05, std=0.018, low=0.015, high=0.14
    ),
    "release_opening_velocity": ParameterPrior(
        mean=0.07, std=0.025, low=0.02, high=0.16
    ),
    # The retreat itself can knock the just-placed cube back down (confirmed via
    # collapsed stack heights despite reported success). 0.08 was safe and 0.14
    # was not, so the upper tail crosses that gap.
    "retract_linear_velocity": ParameterPrior(mean=0.08, std=0.03, low=0.03, high=0.2),
}
"""
Sampling prior for each of :class:`PlaceAction`'s tunable velocity/timing fields.
"""


def build_prior_distribution(
    variables_by_name: Dict[str, Variable], priors: Dict[str, ParameterPrior]
) -> ProbabilisticCircuit:
    """
    Build a factorized Gaussian over ``variables_by_name``, giving every variable whose
    name ends in a field listed in ``priors`` that field's prior.

    ..warning:: :func:`fully_factorized`'s ``variances`` argument is forwarded unchanged
        to ``GaussianDistribution(scale=...)``, which is a standard deviation rather than
        a variance. Each prior's ``std`` is therefore passed as-is; squaring it would
        shrink the spread to ``std**2``, which for these sub-1 values collapses the
        sampling onto the mean.
    """
    means: Dict[Continuous, float] = {}
    standard_deviations: Dict[Continuous, float] = {}
    for name, variable in variables_by_name.items():
        for field_name, prior in priors.items():
            if name.endswith(field_name):
                means[variable] = prior.mean
                standard_deviations[variable] = prior.std
                break

    return fully_factorized(
        means=means,
        variances=standard_deviations,
        variables=variables_by_name.values(),
    )


def _sample_instance(
    match_expr: Match, priors: Dict[str, ParameterPrior], domain_class: Type
):
    """
    Sample one concrete instance from ``match_expr``, whose fields listed in ``priors``
    are left underspecified (``...``) with range where-conditions already applied.

    Builds a factorized Gaussian prior over the underspecified variables (mean/std from
    ``priors``), truncates it to each field's declared range, and draws one sample.

    :param match_expr: The underspecified match, already built via
        ``a(domain_class)(...)`` with the tunable fields set to ``...`` and range
        ``where`` conditions applied.
    :param priors: Maps field name to its sampling prior.
    :param domain_class: The dataclass being sampled (used as the model registry key).
    :return: A fully constructed instance of ``domain_class``.
    """
    match_expr.expression  # populate .variable before UnderspecifiedParameters
    parameters = UnderspecifiedParameters(match_expr)

    distribution = build_prior_distribution(parameters.variables, priors)
    registry = DictRegistry({domain_class: distribution})
    model = registry.get_model(parameters)

    conditioned, _ = model.conditional(
        parameters.conditioning_assignments_from_literal_values
    )
    truncated, _ = conditioned.truncated(
        parameters.truncation_assignments_from_where_conditions
    )
    sample = truncated.sample(1)[0]
    return parameters.construct_instance_from_model_sample(truncated.variables, sample)


def sample_pickup_instance(
    object_body: Body,
    arm: Arms,
    grasp_description: GraspDescription,
    priors: Dict[str, ParameterPrior] = PICKUP_PARAMETER_PRIORS,
) -> PickUpAction:
    """
    Build a :class:`PickUpAction` with its 5 tunable velocity/timing fields plus the
    target object's friction randomly sampled, everything else concrete.

    :param priors: Sampling prior for each tunable field, by field name. Defaults to
        :data:`PICKUP_PARAMETER_PRIORS`; pass a wider set of priors to sample outside
        the range validated to stack reliably.
    """
    match_expr = a(PickUpAction)(
        object_designator=object_body,
        arm=arm,
        # Decomposed field-by-field (matching the pattern used in
        # MoveToReachTrainingEnvironment.setup_plan) rather than passed as one
        # already-built GraspDescription: passing the whole object as a single
        # literal left its enum-typed sub-fields (e.g. vertical_alignment) as
        # Symbolic variables with no domain, since the framework's variable-space
        # walker enumerates a composite field's own sub-fields structurally,
        # independent of whether the field itself was supplied as a literal.
        grasp_description=a(GraspDescription)(
            approach_direction=grasp_description.approach_direction,
            vertical_alignment=grasp_description.vertical_alignment,
            end_effector=grasp_description.end_effector,
            rotate_gripper=grasp_description.rotate_gripper,
            manipulation_offset=grasp_description.manipulation_offset,
        ),
        pre_approach_linear_velocity=...,
        final_approach_linear_velocity=...,
        grasp_closing_velocity=...,
        lift_linear_velocity=...,
        grasp_stall_minimum_time=...,
        object_friction=...,
    )
    match_expr.expression
    v = match_expr.variable
    match_expr.where(
        v.pre_approach_linear_velocity >= priors["pre_approach_linear_velocity"].low,
        v.pre_approach_linear_velocity <= priors["pre_approach_linear_velocity"].high,
        v.final_approach_linear_velocity
        >= priors["final_approach_linear_velocity"].low,
        v.final_approach_linear_velocity
        <= priors["final_approach_linear_velocity"].high,
        v.grasp_closing_velocity >= priors["grasp_closing_velocity"].low,
        v.grasp_closing_velocity <= priors["grasp_closing_velocity"].high,
        v.lift_linear_velocity >= priors["lift_linear_velocity"].low,
        v.lift_linear_velocity <= priors["lift_linear_velocity"].high,
        v.grasp_stall_minimum_time >= priors["grasp_stall_minimum_time"].low,
        v.grasp_stall_minimum_time <= priors["grasp_stall_minimum_time"].high,
        v.object_friction >= priors["object_friction"].low,
        v.object_friction <= priors["object_friction"].high,
    )
    return _sample_instance(match_expr, priors, PickUpAction)


def sample_place_instance(
    object_body: Body,
    target_location: Pose,
    arm: Arms,
    priors: Dict[str, ParameterPrior] = PLACE_PARAMETER_PRIORS,
) -> PlaceAction:
    """
    Build a :class:`PlaceAction` with its 4 tunable velocity fields randomly sampled,
    everything else concrete.

    :param priors: Sampling prior for each tunable field, by field name. Defaults to
        :data:`PLACE_PARAMETER_PRIORS`; pass a wider set of priors to sample outside the
        range validated to stack reliably.
    """
    match_expr = a(PlaceAction)(
        object_designator=object_body,
        target_location=target_location,
        arm=arm,
        transport_linear_velocity=...,
        placing_linear_velocity=...,
        release_opening_velocity=...,
        retract_linear_velocity=...,
    )
    match_expr.expression
    v = match_expr.variable
    match_expr.where(
        v.transport_linear_velocity >= priors["transport_linear_velocity"].low,
        v.transport_linear_velocity <= priors["transport_linear_velocity"].high,
        v.placing_linear_velocity >= priors["placing_linear_velocity"].low,
        v.placing_linear_velocity <= priors["placing_linear_velocity"].high,
        v.release_opening_velocity >= priors["release_opening_velocity"].low,
        v.release_opening_velocity <= priors["release_opening_velocity"].high,
        v.retract_linear_velocity >= priors["retract_linear_velocity"].low,
        v.retract_linear_velocity <= priors["retract_linear_velocity"].high,
    )
    return _sample_instance(match_expr, priors, PlaceAction)
