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
prior (centered on the currently validated defaults), truncated to a safe range. Fitting
a real model on collected successes is a follow-up step once enough varied data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

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
    samples to a safe range -- the Gaussian shape just biases sampling towards values
    near the currently validated default rather than uniformly across the range).
    """

    mean: float
    std: float
    low: float
    high: float


PICKUP_PARAMETER_PRIORS: Dict[str, ParameterPrior] = {
    "pre_approach_linear_velocity": ParameterPrior(
        mean=0.1, std=0.02, low=0.08, high=0.16
    ),
    # Raising this to 0.2 caused a large InfeasibleException failure spike earlier
    # this session (~1/10 success); staying under 0.16 keeps clear margin below
    # that cliff.
    "grasp_linear_velocity": ParameterPrior(mean=0.015, std=0.007, low=0.01, high=0.03),
    # Deliberately slow final approach (avoids shoving the cube before the
    # gripper closes); capped at ~2x default keeps that risk bounded.
    "grasp_closing_velocity": ParameterPrior(mean=0.08, std=0.03, low=0.05, high=0.15),
    # Finger joint physical velocity limit is 0.2 m/s; stays comfortably below it.
    "lift_linear_velocity": ParameterPrior(mean=0.08, std=0.03, low=0.05, high=0.15),
    "grasp_stall_min_time": ParameterPrior(mean=0.4, std=0.15, low=0.3, high=0.8),
    # 0.3 is this session's already-validated floor for stall detection; 0.8
    # stays short of the old fully-conservative 1.0.
    "object_friction": ParameterPrior(mean=1.5, std=0.4, low=0.8, high=2.2),
    # 1.5 (sliding friction) is the cube geoms' own MJCF default; the range
    # stays comfortably above 0 (which would make the object un-graspable)
    # and below values MuJoCo's pyramidal friction cone starts struggling
    # with numerically.
}
"""
Sampling prior for each of :class:`PickUpAction`'s tunable velocity/timing fields.
"""

PLACE_PARAMETER_PRIORS: Dict[str, ParameterPrior] = {
    "transport_linear_velocity": ParameterPrior(
        mean=0.05, std=0.02, low=0.03, high=0.09
    ),
    "placing_linear_velocity": ParameterPrior(
        mean=0.03, std=0.015, low=0.015, high=0.06
    ),
    "release_opening_velocity": ParameterPrior(
        mean=0.05, std=0.02, low=0.03, high=0.09
    ),
    "retract_linear_velocity": ParameterPrior(mean=0.05, std=0.02, low=0.03, high=0.08),
    # Raising this to 0.15 earlier this session caused the retreat itself to
    # knock the just-placed cube back down (confirmed via collapsed stack
    # heights despite reported success); 0.08 stays well short of that.
}
"""
Sampling prior for each of :class:`PlaceAction`'s tunable velocity/timing fields.
"""


def _sample_instance(match_expr: Match, priors: Dict[str, ParameterPrior], domain_class: Type):
    """
    Sample one concrete instance from ``match_expr``, whose fields listed in ``priors``
    are left underspecified (``...``) with range where-conditions already applied.

    Builds a factorized Gaussian prior over the underspecified variables (mean/std from
    ``priors``), truncates it to each field's declared range, and draws one sample.

    :param match_expr: The underspecified match, already built via ``a(domain_class)(...)``
        with the tunable fields set to ``...`` and range ``where`` conditions applied.
    :param priors: Maps field name to its sampling prior.
    :param domain_class: The dataclass being sampled (used as the model registry key).
    :return: A fully constructed instance of ``domain_class``.
    """
    match_expr.expression  # populate .variable before UnderspecifiedParameters
    parameters = UnderspecifiedParameters(match_expr)

    means: Dict[Any, float] = {}
    variances: Dict[Any, float] = {}
    for name, variable in parameters.variables.items():
        for field_name, prior in priors.items():
            if name.endswith(field_name):
                means[variable] = prior.mean
                variances[variable] = prior.std**2
                break

    distribution = fully_factorized(
        means=means, variances=variances, variables=parameters.variables.values()
    )
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
    object_body: Body, arm: Arms, grasp_description: GraspDescription
) -> PickUpAction:
    """
    Build a :class:`PickUpAction` with its 5 tunable velocity/timing fields plus the
    target object's friction randomly sampled (see :data:`PICKUP_PARAMETER_PRIORS`),
    everything else concrete.
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
        grasp_linear_velocity=...,
        grasp_closing_velocity=...,
        lift_linear_velocity=...,
        grasp_stall_min_time=...,
        object_friction=...,
    )
    match_expr.expression
    v = match_expr.variable
    match_expr.where(
        v.pre_approach_linear_velocity >= PICKUP_PARAMETER_PRIORS["pre_approach_linear_velocity"].low,
        v.pre_approach_linear_velocity <= PICKUP_PARAMETER_PRIORS["pre_approach_linear_velocity"].high,
        v.grasp_linear_velocity >= PICKUP_PARAMETER_PRIORS["grasp_linear_velocity"].low,
        v.grasp_linear_velocity <= PICKUP_PARAMETER_PRIORS["grasp_linear_velocity"].high,
        v.grasp_closing_velocity >= PICKUP_PARAMETER_PRIORS["grasp_closing_velocity"].low,
        v.grasp_closing_velocity <= PICKUP_PARAMETER_PRIORS["grasp_closing_velocity"].high,
        v.lift_linear_velocity >= PICKUP_PARAMETER_PRIORS["lift_linear_velocity"].low,
        v.lift_linear_velocity <= PICKUP_PARAMETER_PRIORS["lift_linear_velocity"].high,
        v.grasp_stall_min_time >= PICKUP_PARAMETER_PRIORS["grasp_stall_min_time"].low,
        v.grasp_stall_min_time <= PICKUP_PARAMETER_PRIORS["grasp_stall_min_time"].high,
        v.object_friction >= PICKUP_PARAMETER_PRIORS["object_friction"].low,
        v.object_friction <= PICKUP_PARAMETER_PRIORS["object_friction"].high,
    )
    return _sample_instance(match_expr, PICKUP_PARAMETER_PRIORS, PickUpAction)


def sample_place_instance(object_body: Body, target_location: Pose, arm: Arms) -> PlaceAction:
    """
    Build a :class:`PlaceAction` with its 4 tunable velocity fields randomly sampled
    (see :data:`PLACE_PARAMETER_PRIORS`), everything else concrete.
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
        v.transport_linear_velocity >= PLACE_PARAMETER_PRIORS["transport_linear_velocity"].low,
        v.transport_linear_velocity <= PLACE_PARAMETER_PRIORS["transport_linear_velocity"].high,
        v.placing_linear_velocity >= PLACE_PARAMETER_PRIORS["placing_linear_velocity"].low,
        v.placing_linear_velocity <= PLACE_PARAMETER_PRIORS["placing_linear_velocity"].high,
        v.release_opening_velocity >= PLACE_PARAMETER_PRIORS["release_opening_velocity"].low,
        v.release_opening_velocity <= PLACE_PARAMETER_PRIORS["release_opening_velocity"].high,
        v.retract_linear_velocity >= PLACE_PARAMETER_PRIORS["retract_linear_velocity"].low,
        v.retract_linear_velocity <= PLACE_PARAMETER_PRIORS["retract_linear_velocity"].high,
    )
    return _sample_instance(match_expr, PLACE_PARAMETER_PRIORS, PlaceAction)
