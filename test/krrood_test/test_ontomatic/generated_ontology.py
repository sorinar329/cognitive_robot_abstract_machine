"""
The Python source ontomatic generates from the ontologies of the test dataset, and the module it defines.
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from krrood.ontomatic.ontology_to_python.owl_to_python import OwlToPythonConverter

ONTOLOGIES_DIRECTORY = Path(__file__).parent.parent / "dataset" / "ontologies"
"""
The directory of the ontologies of the test dataset.
"""


@dataclass(kw_only=True)
class GeneratedOntology:
    """
    The files ontomatic generates from an ontology, by their name.
    """

    files: dict[str, str]
    """
    The source of each generated file, by its name.
    """
    module_name: str
    """
    The name of the generated module holding the classes.
    """

    @classmethod
    def generate_from_file(cls, *, ontology_file_name: str) -> GeneratedOntology:
        """
        :param ontology_file_name: The name of an ontology file of the test dataset.
        :return: The files generated from it, named after the ontology.
        """
        converter = OwlToPythonConverter()
        converter.load_ontology(path=str(ONTOLOGIES_DIRECTORY / ontology_file_name))
        module_name = Path(ontology_file_name).stem
        return cls(
            files=converter.generate_python_code_external(base_file_name=module_name),
            module_name=module_name,
        )

    @property
    def classes_source(self) -> str:
        """
        The source of the generated module holding the classes.
        """
        return self.files[f"{self.module_name}.py"]

    def class_source(self, *, class_name: str) -> str:
        """
        :return: The source of the generated class, from its ``class`` line to the next class.
        """
        match = re.search(
            rf"^class {class_name}\(.*?(?=^@dataclass|^class |\Z)",
            self.classes_source,
            re.S | re.M,
        )
        assert match, f"class {class_name} not generated"
        return match.group(0)

    def method_source(self, *, class_name: str, method_name: str) -> str:
        """
        :return: The source of a method of a generated class, up to the next method.
        """
        class_source = self.class_source(class_name=class_name)
        start = class_source.index(f"def {method_name}(")
        following = class_source.find("@classmethod", start)
        return class_source[start : following if following != -1 else None]

    def imported(self, *, root: Path) -> ModuleType:
        """
        Write the files as a package in the root directory and import its module holding the classes.

        :param root: An empty directory.
        :return: The imported module.
        """
        package = root / f"generated_{self.module_name}"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for file_name, source in self.files.items():
            (package / file_name).write_text(source)
        sys.path.insert(0, str(root))
        return importlib.import_module(f"{package.name}.{self.module_name}")
