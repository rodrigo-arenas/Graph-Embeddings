import networkx as nx
from stellargraph import StellarGraph


def get_stellar_graph(graph):
    if isinstance(graph, nx.classes.graph.Graph):
        stellar_graph = StellarGraph.from_networkx(graph)
    elif isinstance(graph, StellarGraph):
        stellar_graph = graph
    else:
        raise ValueError("graph should be an instance of networkx or StellarGraph object")

    return stellar_graph


def get_stellar_graph_list(graphs):

    return [get_stellar_graph(graph) for graph in graphs]
