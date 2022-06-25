# Graph-Embeddings
Graph embeddings for downstream tasks

# Algorithms:

## StackedNode2Vec

Computes the Node2Vec representation of each node in a set of graphs.

### Example:

```python
import networkx as nx
from graph_embeddings.algorithms import StackedNode2Vec

g1 = nx.DiGraph()
g2 = nx.DiGraph()
g1.add_edges_from([("A", "B"), ("B", "C"), ("C", "B"), ("B", "E")])
g2.add_edges_from([("A", "B"), ("B", "C"), ("C", "B"), ("B", "E")])

graphs = [g1, g2]
embedding_model = StackedNode2Vec()
embedding_model.fit(graphs)

embedding_model.get_embeddings()  # ndarray with dimensions (4, 128, 2)
embedding_model.get_dense_embeddings()  # ndarray with dimensions (2, 512)

```

Changelog
#########

See the `changelog <https://graph-embeddings.readthedocs.io/en/latest/release_notes.html>`__
for notes on the changes of graph-embeddings

Important links
###############

- Official source code repo: https://github.com/rodrigo-arenas/graph-embeddings/
- Download releases: https://pypi.org/project/graph-embeddings/
- Issue tracker: https://github.com/rodrigo-arenas/graph-embeddings/issues
- Stable documentation: https://graph-embeddings.readthedocs.io/en/stable/

Source code
###########

You can check the latest development version with the command::

   git clone https://github.com/rodrigo-arenas/graph-embeddings.git

Install the development dependencies::
  
  pip install -r dev-requirements.txt
  
Check the latest in-development documentation: https://graph-embeddings.readthedocs.io/en/latest/

Contributing
############

Contributions are more than welcome!
There are several opportunities on the ongoing project, so please get in touch if you would like to help out.
Make sure to check the current issues and also
the `Contribution guide <https://github.com/rodrigo-arenas/graph-embeddings/blob/master/CONTRIBUTING.md>`_.



