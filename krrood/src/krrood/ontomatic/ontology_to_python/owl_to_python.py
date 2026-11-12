"""
Converting an OWL ontology into Python source code.
"""

from __future__ import annotations

import os

import rdflib
from typing_extensions import Dict, Optional


from krrood.ontomatic.ontology_to_python.code_generation import CodeGenerator
from krrood.ontomatic.ontology_to_python.extraction import (
    OntologyExtractor,
)
from krrood.ontomatic.ontology_to_python.ontology_info import (
    OntologyInfo,
)


class OwlToPythonConverter:
    """
    High-level converter for transforming an OWL ontology into Python source code.
    """

    def __init__(
        self, predefined_data_types: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the converter.

        :param predefined_data_types: Manual type overrides for properties.
        """
        self.graph = rdflib.Graph()
        self.predefined_data_types = predefined_data_types or {}
        self.ontology_info: Optional[OntologyInfo] = None

    def load_ontology(self, *, path: str):
        """
        Load an OWL ontology from a file.

        :param path: Path to the .owl file.
        """
        self.graph.parse(path)
        self.ontology_info = OntologyExtractor(
            graph=self.graph, predefined_data_types=self.predefined_data_types
        ).extract()

    def generate_python_code_external(self, *, base_file_name: str) -> Dict[str, str]:
        """
        Generate Python code without saving to disk.

        :param base_file_name: Base name for the generated files.
        :return: Dictionary of filename to content.
        """
        generator = CodeGenerator(ontology=self.ontology_info)
        return generator.generate(base_file_name=base_file_name)

    def save_to_file(self, *, output_path: str):
        """
        Generate Python code and save it to the specified output path.

        :param output_path: Base path (filename) for the main output file.
        """
        base = os.path.splitext(os.path.basename(output_path))[0]
        for name, content in self.generate_python_code_external(
            base_file_name=base
        ).items():
            with open(
                os.path.join(os.path.dirname(output_path), name), "w", encoding="utf-8"
            ) as file:
                file.write(content)
