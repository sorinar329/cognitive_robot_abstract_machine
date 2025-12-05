import pytest

from krrood.entity_query_language.symbol_graph import SymbolGraph


@pytest.fixture(autouse=True)
def cleanup_after_test():
    # Setup: runs before each krrood_test
    SymbolGraph()
    yield
    SymbolGraph().clear()