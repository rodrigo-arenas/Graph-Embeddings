import pytest
import networkx as nx
import numpy as np
from stellargraph import StellarGraph
from ..node2vec import StackedNode2Vec

g1 = nx.DiGraph()
g2 = nx.DiGraph()
g1.add_edges_from([("A", "B"), ("B", "C"), ("C", "B"), ("B", "E")])
g2.add_edges_from([("A", "B"), ("B", "D"), ("D", "C"), ("C", "D")])

g_nx = [g1, g2]
g_stellar = [StellarGraph.from_networkx(g) for g in g_nx]


@pytest.mark.parametrize(
    "graph",
    [g_nx, g_stellar],
)
def test_node2vec_properties(graph):
    embeddings_model = StackedNode2Vec(node_embeddings_size=16)
    embeddings_model.fit(graph)

    assert len(embeddings_model.unique_nodes) == 5
    assert embeddings_model.embeddings.shape == (5, 16, 2)  # nodes, embedding_size, graphs
    assert embeddings_model.dense_embeddings.shape == (2, 16 * 5)  # graphs, embedding_size*nodes
    assert np.array_equal(embeddings_model.embeddings, embeddings_model.get_embeddings())
    assert np.array_equal(embeddings_model.dense_embeddings, embeddings_model.get_dense_embeddings())

