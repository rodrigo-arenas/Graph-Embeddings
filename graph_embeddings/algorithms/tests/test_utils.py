import pytest
import networkx as nx
from stellargraph import StellarGraph
from ..utils import get_stellar_graph, get_stellar_graph_list


g1 = nx.DiGraph()
g2 = nx.DiGraph()
g1.add_edges_from([("A", "B"), ("B", "C"), ("C", "B"), ("B", "E")])
g2.add_edges_from([("A", "B"), ("B", "D"), ("D", "C"), ("C", "D")])
g3 = StellarGraph.from_networkx(g2)

g_nx = [g1, g2]
g_mixed = [g1, g3]


@pytest.mark.parametrize(
    "graph",
    [g1, g3],
)
def test_get_stellar_graph(graph):
    assert isinstance(get_stellar_graph(graph), StellarGraph)

    with pytest.raises(Exception) as excinfo:
        get_stellar_graph(6)

    assert (
            str(excinfo.value) == "graph should be an instance of networkx or StellarGraph object"
    )


@pytest.mark.parametrize(
    "graph_list",
    [g_nx, g_mixed],
)
def test_get_stellar_graph_list(graph_list):
    for graph_instance in get_stellar_graph_list(graph_list):
        assert isinstance(graph_instance, StellarGraph)

