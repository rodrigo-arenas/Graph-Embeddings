import sys
import itertools
import networkx as nx
import numpy as np
from typing import List, Union
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from tqdm.auto import tqdm


class StackedNode2Vec:
    def __init__(self, node_embeddings_size=128, padding_value=0, window=5,
                 walk_length: int = 150, n=25, p=0.5, q=2.0, sg=1, n_jobs: int = 1):
        """
        Computes the Node2Vec representation of each node in a set of graphs.

        Parameters
        ----------
        node_embeddings_size: int, default= 128
            Size of the Node2Vec vector of each node
        padding_value: int, default=0
            This number fills the rows in the graph's matrix,
            corresponding to a node that is not present in an individual graph
        walk_length: int, default=150
            Maximum length of each random walk
        window: int, default=5
            Maximum distance between the current and predicted word within a sentence.
        n: int, default=25
            Total number of random walks per root node
        p: float, default=0.5
            Defines probability, 1/p, of returning to source node
        q: float, default=2.0
            Defines probability, 1/q, for moving to a node away from the source node
        sg: {0, 1}
            Training algorithm: 1 for skip-gram; otherwise CBOW.
        n_jobs: int, default=1
            Use these many worker threads to train the model (=faster training with multicore machines).
        """
        self.node_embeddings_size = node_embeddings_size
        self.padding_value = padding_value
        self.walk_length = walk_length
        self.window = window
        self.n = n
        self.p = p
        self.q = q
        self.sg = sg
        self.n_jobs = n_jobs

        self.unique_nodes = None
        self.embeddings = None
        self.dense_embeddings = None

    def fit(self, graphs: List[Union[nx.classes.graph.Graph, StellarGraph]]):
        """
        Parameters
        ----------
        graphs: List[Union[nx.classes.graph.Graph, StellarGraph]]
            List of Networkx or StellarGraph objects
        Returns
        -------
        self: object
        """

        nodes = [set(graph.nodes()) for graph in graphs]
        self.unique_nodes = list(set(itertools.chain(*nodes)))
        n_nodes = len(self.unique_nodes)
        n_graphs = len(graphs)
        nodes_to_id = {node: i for i, node in enumerate(self.unique_nodes)}

        self.dense_embeddings = np.empty(shape=(n_graphs, n_nodes * self.node_embeddings_size))
        self.embeddings = np.empty(shape=(n_nodes, self.node_embeddings_size, n_graphs))

        progress_bar = tqdm(file=sys.stdout, total=n_graphs)
        for i, graph in enumerate(graphs):
            progress_bar.update(1)
            embeddings = np.empty(shape=(n_nodes, self.node_embeddings_size))
            embeddings.fill(self.padding_value)
            if isinstance(graph, nx.classes.graph.Graph):
                stellar_graph = StellarGraph.from_networkx(graph)
            else:
                stellar_graph = graph

            graph_nodes = list(stellar_graph.nodes())
            random_walk = BiasedRandomWalk(stellar_graph)

            walks = random_walk.run(
                nodes=graph_nodes,
                length=self.walk_length,
                n=self.n,
                p=self.p,
                q=self.q,
            )

            str_walks = [[str(n) for n in walk] for walk in walks]
            model = Word2Vec(str_walks,
                             vector_size=self.node_embeddings_size,
                             window=self.window,
                             min_count=0,
                             sg=self.sg,
                             workers=self.n_jobs)

            for j, node in enumerate(graph_nodes):
                node_idx = nodes_to_id[node]
                embeddings[node_idx] = np.array(model.wv.vectors[j])

            self.embeddings[:, :, i] = embeddings
            self.dense_embeddings[i] = embeddings.flatten()

        progress_bar.close()

        return self

    def get_embeddings(self):
        """
        Returns
        -------
        embeddings: np.ndarray
            Array with dimensions (n_nodes, node_embeddings_size, n_graphs) representing the
            Node2Vec result of each node in each graph
        """
        return self.embeddings

    def get_dense_embeddings(self):
        """
        Returns
        -------
        embeddings: np.ndarray
            Array with dimensions (n_graphs, n_nodes*node_embeddings_size, ) representing the
            flatten Node2Vec result of each node in each graph
        """
        return self.dense_embeddings
