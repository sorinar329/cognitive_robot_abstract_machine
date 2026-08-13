import unittest
import sys

from PySide6.QtWidgets import QWidget, QApplication

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from probabilistic_model.gui.home_widget import HomeWidget
from probabilistic_model.gui.main_window import MainWindow
from probabilistic_model.gui.mode_widget import ModeWidget
from probabilistic_model.gui.plotting import (
    InteractiveChartView,
    ProbabilisticModelPlotWidget,
)
from probabilistic_model.gui.posterior_widget import PosteriorWidget
from probabilistic_model.gui.query_widget import QueryWidget
from probabilistic_model.gui.variable_constraint_widget import VariableConstraintWidget
from random_events.variable import Continuous


class TestDataclassParentFieldDoesNotShadowQWidgetParent(unittest.TestCase):
    """
    Every page widget takes its Qt parent through a dataclass ``InitVar`` for
    :meth:`QWidget.__init__`.

    Naming that field ``parent`` makes ``@dataclass`` set a class attribute of the same
    name to its default value, which shadows the inherited ``QWidget.parent`` *method*
    with ``None`` for every instance.
    """

    def test_widget_classes_do_not_shadow_parent_method(self):
        for widget_class in (
            HomeWidget,
            MainWindow,
            ModeWidget,
            InteractiveChartView,
            ProbabilisticModelPlotWidget,
            PosteriorWidget,
            QueryWidget,
        ):
            self.assertNotEqual(
                getattr(widget_class, "parent", None),
                None,
                f"{widget_class.__name__}.parent must stay QWidget.parent, not a "
                "dataclass-shadowed None",
            )


class TestNumericIntervalWidgetCreation(unittest.TestCase):
    """
    Selecting a numeric variable in a fresh :class:`VariableConstraintWidget` builds a
    :class:`~probabilistic_model.gui.variable_constraint_widget.NumericIntervalWidget`
    with a range slider.

    Building it walks the slider's ancestor chain (superqt's
    ``update_styles_from_stylesheet``) calling ``.parent()`` on each ancestor -- which
    raises ``TypeError: 'NoneType' object is not callable`` if any ancestor's
    ``.parent`` is dataclass-shadowed, since the walk then calls ``None()`` instead of
    ``QWidget.parent()``.
    """

    def test_selecting_a_continuous_variable_builds_a_range_slider(self):
        variable = Continuous(name="v1")
        widget = VariableConstraintWidget([variable])
        widget.variable_combo.setCurrentIndex(1)  # index 0 is "Select Variable..."

        self.assertEqual(len(widget.interval_widgets), 1)
        self.assertIsNotNone(widget.interval_widgets[0].slider)


if __name__ == "__main__":
    unittest.main()
