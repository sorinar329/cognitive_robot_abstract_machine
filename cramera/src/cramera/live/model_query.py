"""
Live evidence-conditioned posterior queries against a trained causal-diagnosis model.

Loads the same ``<name>_circuit.json`` files :mod:`export_posterior_plots` reads (a
:class:`ProbabilisticCircuit`, not the :class:`JointProbabilityTree` it was fit as), and
computes the distribution of one or more query variables after conditioning on evidence
-- the same operation the desktop GUI's Posterior page performs. A model is fit once and
does not change while a demo runs, so this has no live bridge or scene dependency: any
cramera process can answer these queries independently of whether a demo is running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from typing_extensions import Dict, List

from probabilistic_model.gui.distribution_payload import (
    numeric_variable_distribution_payload,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


class UnknownModelVariable(Exception):
    """
    Raised when a requested query or evidence variable does not exist in the model.
    """


class EvidenceHasZeroProbability(Exception):
    """
    Raised when the requested evidence conditions out the entire model.
    """


@dataclass
class EvidenceConstraint:
    """
    One evidence row: a closed interval to condition a numeric variable on.
    """

    variable: str
    """
    Name of the variable this constraint conditions.
    """

    minimum: float
    """
    Lower bound of the evidence interval, inclusive.
    """

    maximum: float
    """
    Upper bound of the evidence interval, inclusive.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> EvidenceConstraint:
        """
        Build a constraint from one entry of a request's ``evidence`` list.

        :param payload: A dict with ``variable``, ``minimum`` and ``maximum`` keys.
        """
        return cls(
            variable=str(payload["variable"]),
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
        )


@dataclass
class ModelQueryService:
    """
    Answers posterior queries against the causal-diagnosis models fit for this demo.
    """

    models_directory: Path
    """
    Directory containing ``<name>_circuit.json`` files.
    """

    _circuits: Dict[str, ProbabilisticCircuit] = field(default_factory=dict, init=False)
    """
    Loaded circuits, cached by model name since a circuit does not change once fit.
    """

    def _circuit(self, model_name: str) -> ProbabilisticCircuit:
        """
        The loaded circuit for ``model_name``, loading and caching it on first use.

        :param model_name: Which ``<model_name>_circuit.json`` to load.
        """
        if model_name not in self._circuits:
            circuit_path = self.models_directory / f"{model_name}_circuit.json"
            with circuit_path.open() as circuit_file:
                self._circuits[model_name] = ProbabilisticCircuit.from_json(
                    json.load(circuit_file)
                )
        return self._circuits[model_name]

    def posterior(
        self,
        model_name: str,
        evidence: List[EvidenceConstraint],
        query_variables: List[str],
    ) -> Dict[str, dict]:
        """
        Distribution data for each of ``query_variables``, conditioned on ``evidence``.

        :param model_name: Which model to query, e.g. ``"pickup"``.
        :param evidence: Constraints to condition the model on before marginalizing.
        :param query_variables: Names of the variables to compute the distribution of.
        :return: A mapping from variable name to the same payload shape
            :func:`~probabilistic_model.gui.distribution_payload.numeric_variable_distribution_payload`
            returns.
        """
        circuit = self._circuit(model_name)
        variable_map = {variable.name: variable for variable in circuit.variables}

        model = circuit
        if evidence:
            evidence_event = self._build_event(evidence, variable_map)
            model, probability = circuit.truncated(evidence_event)
            if probability == 0 or model is None:
                raise EvidenceHasZeroProbability(
                    f"evidence {evidence} has zero probability under {model_name!r}"
                )

        payloads = {}
        for name in query_variables:
            variable = self._require_variable(name, variable_map)
            payloads[name] = numeric_variable_distribution_payload(model, variable)
        return payloads

    def _build_event(
        self,
        evidence: List[EvidenceConstraint],
        variable_map: Dict[str, Variable],
    ) -> Event:
        """
        The conjunctive event ``evidence``'s constraints describe.

        :param evidence: Constraints to combine, one interval per variable.
        :param variable_map: Variables of the model being queried, by name.
        """
        simple_event = SimpleEvent.from_data()
        for constraint in evidence:
            variable = self._require_variable(constraint.variable, variable_map)
            simple_event[variable] = closed(constraint.minimum, constraint.maximum)
        return Event.from_simple_sets(simple_event)

    @staticmethod
    def _require_variable(name: str, variable_map: Dict[str, Variable]) -> Variable:
        """
        The model variable named ``name``.

        :param name: Name of the variable to look up.
        :param variable_map: Variables of the model being queried, by name.
        """
        if name not in variable_map:
            raise UnknownModelVariable(name)
        return variable_map[name]
