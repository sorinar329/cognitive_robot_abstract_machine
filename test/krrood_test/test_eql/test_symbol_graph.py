import gc
import os
import time
import weakref

import pytest

from krrood.entity_query_language.factories import entity, variable, an
from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.example_classes import KRROODPosition

try:
    import pydot
    import pygraphviz
except ImportError:
    pydot = None
    pygraphviz = None


@pytest.mark.skipif(
    not (pydot and pygraphviz), reason="pydot and graphviz not installed"
)
def test_visualize_symbol_graph():
    SymbolGraph().clear()
    symbol_graph = SymbolGraph()
    symbol_graph.to_dot("symbol_graph.svg", format_="svg", graph_type="type")
    assert len(symbol_graph._class_diagram.wrapped_classes) >= 59
    if os.path.exists("symbol_graph.svg"):
        os.remove("symbol_graph.svg")


def test_memory_leak():
    """
    Test if the SymbolGraph does not artificially keep objects alive that would be
    garbage collected.
    """

    def create_data():
        point = KRROODPosition(1, 2, 3)
        return point

    create_data()

    q = an(entity(variable(KRROODPosition, domain=None)))
    result = list(q.evaluate())

    assert result == []

    assert len(SymbolGraph().wrapped_instances) == 0


# %% forgetting symbols that died


def test_the_symbol_graph_lets_go_of_a_dead_symbols_wrapper():
    """
    Once a symbol has died and the graph removed it, nothing in the graph keeps its
    wrapper alive.
    """
    symbol_graph = SymbolGraph()
    position = KRROODPosition(1, 2, 3)
    wrapper = weakref.ref(symbol_graph.get_wrapped_instance(position))
    del position

    symbol_graph.remove_dead_instances()
    gc.collect()

    assert wrapper() is None


def test_removing_dead_symbols_takes_time_in_proportion_to_their_number():
    """
    Removing twice as many dead symbols, among twice as many living ones, takes about
    twice as long, not four times.
    """

    def seconds_to_remove(count: int) -> float:
        SymbolGraph.clear()
        symbol_graph = SymbolGraph()
        living = [KRROODPosition(0, 0, 0) for _ in range(count)]
        dying = [KRROODPosition(1, 1, 1) for _ in range(count)]
        del dying
        start = time.perf_counter()
        symbol_graph.remove_dead_instances()
        seconds = time.perf_counter() - start
        assert len(living) == count
        return seconds

    smaller_round = 4000
    seconds_to_remove(smaller_round)  # warm up
    assert seconds_to_remove(2 * smaller_round) < 3 * seconds_to_remove(smaller_round)
