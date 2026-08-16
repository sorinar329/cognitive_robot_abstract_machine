"""
Third-generation copy of ``export_posterior_plots.py``, exporting
``training/models_v3/<name>_circuit.json`` -- ``train_jpt3.py``'s fitted circuits --
for the cramera live viewer's Posterior tab, instead of ``train_jpt2.py``'s
``models_v2``.

Publishing is not versioned by generation: the exported files land at the same
``<name>.json`` / ``<name>_circuit.json`` names ``export_posterior_plots.py`` uses, so
running this after that script (or vice versa) replaces whichever generation's models
were last published -- only one demo's models are "live" for the viewer at a time.

Run it with the interpreter whose packages point at this checkout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from probabilistic_model.gui.distribution_payload import (
    numeric_variable_distribution_payload,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from causal_diagnosis_v3 import MODEL_DIRECTORY

from cramera import paths as cramera_paths

MODEL_NAMES: tuple[str, ...] = ("pickup", "place")
"""
Which ``models_v3/<name>_circuit.json`` files to export.
"""

WEB_DATA_DIRECTORY = (
    Path(__file__).resolve().parents[4]
    / "cramera"
    / "src"
    / "cramera"
    / "web"
    / "data"
    / "posterior"
)
"""
Where the exported ``<name>.json`` files are written -- inside the cramera package's own
static web assets, so the frontend can fetch them with no backend endpoint of their own.
"""


def export_posterior(model_name: str) -> Path:
    """
    Export every numeric variable's posterior distribution for
    ``models_v3/<model_name>_circuit.json``.

    :param model_name: Which model to export, e.g. ``"pickup"``.
    :return: Path the exported JSON was written to.
    """
    circuit_path = MODEL_DIRECTORY / f"{model_name}_circuit.json"
    with circuit_path.open() as circuit_file:
        circuit = ProbabilisticCircuit.from_json(json.load(circuit_file))

    live_models_directory = cramera_paths.models_directory()
    live_models_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(circuit_path, live_models_directory / circuit_path.name)

    variables = {}
    for variable in sorted(circuit.variables, key=lambda v: v.name):
        if not variable.is_numeric:
            # none of this demo's variables are symbolic today; skipped rather than
            # guessed at, since the viewer has nothing built for that case yet
            continue
        variables[variable.name] = numeric_variable_distribution_payload(
            circuit, variable
        )

    WEB_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = WEB_DATA_DIRECTORY / f"{model_name}.json"
    with output_path.open("w") as output_file:
        json.dump(
            {"order": list(variables.keys()), "variables": variables}, output_file
        )
    print(f"wrote {output_path} ({len(variables)} variables)")
    return output_path


def main() -> None:
    for model_name in MODEL_NAMES:
        export_posterior(model_name)


if __name__ == "__main__":
    main()
