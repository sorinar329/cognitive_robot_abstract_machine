"""
Third-generation copy of ``train_jpt2.py``, training on
``export_successful_parameters_v3.py``'s CSVs -- this generation's own separate
collection run, not the ``coraplex_panda_demo_v2`` archive -- and writing the fitted
trees under :data:`MODEL_DIRECTORY` so they do not overwrite ``train_jpt2.py``'s own
models.

One tree is learned per action -- ``PickUpAction``'s and ``PlaceAction``'s sampled
fields are unrelated variables from two different action classes, so a single tree
spanning both would not describe either action's own distribution (see
``export_successful_parameters_v3.py``'s ``ACTION_PARAMETER_COLUMNS``). Each tree is
learned over its action's parameters together with ``step_index`` and
``object_final_z``, so it can both tell which of the three stacking steps a row describes
and support the causal diagnosis in ``causal_diagnosis_v3.py``, which uses
``object_final_z`` as the effect variable in place of ``step_index``.

Every tree is validated on data it was not fit on before it is saved -- see
:func:`held_out_validation`.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python train_jpt3.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy
import pandas

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

# %% where the data comes from and the models go

DATASET_DIRECTORY = Path(__file__).parent.parent / "datasets_v3"
"""
Where ``export_successful_parameters_v3.py`` wrote the learning CSVs.
"""

MODEL_DIRECTORY = Path(__file__).parent / "models_v3"
"""
Where a fitted action's tree is written, as JSON.
"""

ACTION_CSV_NAMES: dict[str, str] = {
    "pickup": "pickup_parameters.csv",
    "place": "place_parameters.csv",
}
"""
Which learning CSV holds each action's sampled parameters.
"""

MIN_SAMPLES_PER_LEAF = 0.03
"""
Minimum fraction of an action's attempts a leaf must cover before the tree is allowed
to add another top-level decision split.

Chosen coarse on purpose: the tree is a prior to sample new parameter combinations
from, not a lookup table of the training attempts, so leaves wide enough to average
over dozens of attempts generalize better than ones fit to a handful of points.

..note:: On ``train_jpt2.py``'s data, this bound was never actually the binding one --
    the sampled parameters are close to independent of each other, so
    :class:`JointProbabilityTree` stops adding decision splits (finds no more impurity
    gain) long before hitting this floor regardless of its value. Re-check this note
    against this generation's own data once it is fit; :data:`MIN_SAMPLES_PER_QUANTILE`
    is what actually controls how many pieces each leaf's own per-variable distribution
    is fit with.
"""

MIN_SAMPLES_PER_QUANTILE = 100
"""
Minimum number of an action's attempts each piece of a leaf's per-variable continuous
distribution must cover (passed to :func:`infer_variables_from_dataframe`, which forwards
it to each variable's :class:`~probabilistic_model.learning.nyga_induction.NygaInduction`
fit).

Carried over from ``train_jpt2.py``'s own tuning, found there via a held-out
log-likelihood sweep from 10 to 150 against ``leaf`` count -- see that module's own note
for the numbers. Re-sweep against this generation's own data if held-out validation here
looks materially different.
"""

MIN_LIKELIHOOD_IMPROVEMENT = 0.1
"""
Minimum relative likelihood improvement a further split of a leaf's per-variable
continuous distribution must provide (passed to :func:`infer_variables_from_dataframe`
the same way as :data:`MIN_SAMPLES_PER_QUANTILE`).

Left at the library default, carried over from ``train_jpt2.py``: sweeping it from 0.01
to 0.3 changed nothing on that generation's data once
:data:`MIN_SAMPLES_PER_QUANTILE` was the binding constraint.
"""

HOLDOUT_FRACTION = 0.1
"""
Fraction of an action's attempts withheld from fitting, to check the fitted tree
assigns a reasonable likelihood to attempts it never saw.
"""

RANDOM_STATE = 0
"""
Seed the train/holdout split is drawn with, so a validation run is reproducible.
"""

MAX_ZERO_LIKELIHOOD_FRACTION = 0.05
"""
Largest fraction of withheld attempts allowed to get zero likelihood before
:class:`DegenerateHeldOutLikelihood` is raised.
"""


# %% fitting


def fit_tree(
    data: pandas.DataFrame,
) -> tuple[JointProbabilityTree, ProbabilisticCircuit]:
    """
    Learn a tree over every column of ``data``.

    :param data: One action's sampled parameters, plus ``step_index`` and
        ``object_final_z``.
    """
    variables = infer_variables_from_dataframe(
        data,
        min_samples_per_quantile=MIN_SAMPLES_PER_QUANTILE,
        min_likelihood_improvement=MIN_LIKELIHOOD_IMPROVEMENT,
    )
    tree = JointProbabilityTree(
        annotated_variables=variables, min_samples_per_leaf=MIN_SAMPLES_PER_LEAF
    )
    circuit = tree.fit(data)
    return tree, circuit


def as_variable_ordered_array(
    tree: JointProbabilityTree, data: pandas.DataFrame
) -> numpy.ndarray:
    """
    :param tree: The tree whose column order ``data`` must be aligned to.
    :param data: Data with the same columns as ``tree`` was fit on, in any order.
    :return: ``data`` as a numpy array, with columns reordered to match
        ``tree.variables`` -- the order its circuit's ``likelihood`` expects, which is
        not necessarily the order the columns were given in.
    """
    return data[[variable.name for variable in tree.variables]].to_numpy()


# %% validating


@dataclass
class HeldOutValidationResult:
    """
    How well a tree fit on a training split explains the attempts withheld from it.
    """

    mean_log_likelihood: float
    """
    Mean log-likelihood of the withheld attempts that did not get zero likelihood.
    """

    zero_likelihood_count: int
    """
    How many withheld attempts the fitted tree assigned zero likelihood.
    """

    holdout_size: int
    """
    How many attempts were withheld.
    """


class DegenerateHeldOutLikelihood(RuntimeError):
    """
    Raised when a fitted tree assigns zero likelihood to too many withheld attempts.

    A tree this disconnected from unseen data would sample parameter combinations the
    training data never actually supports, which is what this check exists to catch
    before a model reaches :data:`MODEL_DIRECTORY`.
    """

    def __init__(self, action: str, result: HeldOutValidationResult) -> None:
        super().__init__(
            f"{action}: {result.zero_likelihood_count}/{result.holdout_size} withheld "
            "attempts got zero likelihood"
        )


class NonPositiveTrainingLikelihood(RuntimeError):
    """
    Raised when a fitted tree assigns zero likelihood to its own training attempts.

    A tree should always cover the data it was fit on; failing this means the fit
    itself is broken, not just imprecise.
    """

    def __init__(self, action: str, zero_count: int) -> None:
        super().__init__(
            f"{action}: tree assigns zero likelihood to {zero_count} of its own "
            "training attempts"
        )


class RoundTripMismatch(RuntimeError):
    """
    Raised when a fitted tree differs from itself after a JSON round-trip.

    The saved model is only useful if loading it back reproduces the tree that was
    actually validated above, not a close approximation of it.
    """

    def __init__(self, action: str) -> None:
        super().__init__(
            f"{action}: tree changed across a to_json/from_json round-trip"
        )


def held_out_validation(
    data: pandas.DataFrame, holdout_fraction: float, random_state: int
) -> HeldOutValidationResult:
    """
    Fit a tree on a random subset of ``data`` and score the rest under it.

    :param data: One action's sampled parameters, plus ``step_index`` and
        ``object_final_z``.
    :param holdout_fraction: Fraction of ``data`` withheld from fitting.
    :param random_state: Seed the train/holdout split is drawn with.
    """
    holdout = data.sample(frac=holdout_fraction, random_state=random_state)
    train = data.drop(holdout.index)

    tree, circuit = fit_tree(train)
    likelihoods = circuit.likelihood(as_variable_ordered_array(tree, holdout))

    positive = likelihoods[likelihoods > 0]
    zero_count = int(len(likelihoods) - len(positive))
    mean_log_likelihood = (
        float(numpy.mean(numpy.log(positive))) if len(positive) else float("-inf")
    )
    return HeldOutValidationResult(mean_log_likelihood, zero_count, len(holdout))


def validate_training_likelihood(
    action: str,
    tree: JointProbabilityTree,
    circuit: ProbabilisticCircuit,
    data: pandas.DataFrame,
) -> None:
    """
    Confirm a fitted tree assigns positive likelihood to every attempt it was fit on.

    :param action: Name of the action the tree is for, for the error message.
    :param tree: The fitted tree, for its column order.
    :param circuit: The fitted tree's circuit.
    :param data: The data the tree was fit on.
    :raises NonPositiveTrainingLikelihood: If any attempt gets zero likelihood.
    """
    likelihoods = circuit.likelihood(as_variable_ordered_array(tree, data))
    zero_count = int((likelihoods <= 0).sum())
    if zero_count:
        raise NonPositiveTrainingLikelihood(action, zero_count)


def validate_round_trip(action: str, tree: JointProbabilityTree) -> None:
    """
    Confirm a fitted tree survives a JSON round-trip unchanged.

    :param action: Name of the action the tree is for, for the error message.
    :param tree: The fitted tree to check.
    :raises RoundTripMismatch: If the round-tripped tree differs from the original.
    """
    restored = from_json(to_json(tree))
    if restored != tree:
        raise RoundTripMismatch(action)


# %% running the pipeline


def train_action(action: str, csv_name: str) -> None:
    """
    Fit, validate and save one action's tree.

    :param action: Name of the action the tree is for, e.g. ``"pickup"``.
    :param csv_name: File under :data:`DATASET_DIRECTORY` holding its parameters.
    """
    data = pandas.read_csv(DATASET_DIRECTORY / csv_name)

    result = held_out_validation(data, HOLDOUT_FRACTION, RANDOM_STATE)
    if (
        result.zero_likelihood_count / result.holdout_size
        > MAX_ZERO_LIKELIHOOD_FRACTION
    ):
        raise DegenerateHeldOutLikelihood(action, result)
    print(
        f"[{action}] held-out validation: mean log-likelihood "
        f"{result.mean_log_likelihood:.3f} over {result.holdout_size} attempts "
        f"({result.zero_likelihood_count} at zero likelihood)"
    )

    tree, circuit = fit_tree(data)
    validate_training_likelihood(action, tree, circuit, data)
    validate_round_trip(action, tree)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIRECTORY / f"{action}_jpt.json"
    with model_path.open("w") as model_file:
        json.dump(to_json(tree), model_file)
    print(f"[{action}] {model_path.name}: fit on {len(data)} attempts, tree saved")


def main() -> None:
    for action, csv_name in sorted(ACTION_CSV_NAMES.items()):
        train_action(action, csv_name)


if __name__ == "__main__":
    main()
