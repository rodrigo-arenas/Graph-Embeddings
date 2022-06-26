.. -*- mode: rst -*-

|Tests|_ |Codecov|_ |PythonVersion|_ |PyPi|_ |Docs|_

.. |Tests| image:: https://github.com/rodrigo-arenas/graph-embeddings/actions/workflows/ci-tests.yml/badge.svg?branch=main
.. _Tests: https://github.com/rodrigo-arenas/Graph-Embeddings/actions/workflows/ci-tests.yml

.. |Codecov| image:: https://codecov.io/gh/rodrigo-arenas/graph-embeddings/branch/main/graphs/badge.svg?branch=main&service=github
.. _Codecov: https://codecov.io/github/rodrigo-arenas/graph-embeddings?branch=main

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8-blue
.. _PythonVersion : https://www.python.org/downloads/

.. |PyPi| image:: https://badge.fury.io/py/graph-embeddings.svg
.. _PyPi: https://badge.fury.io/py/graph-embeddings

.. |Docs| image:: https://readthedocs.org/projects/graph-embeddings/badge/?version=latest
.. _Docs: https://graph-embeddings.readthedocs.io/en/latest/?badge=latest

.. |Contributors| image:: https://contributors-img.web.app/image?repo=rodrigo-arenas/graph-embeddings
.. _Contributors: https://github.com/rodrigo-arenas/Graph-Embeddings/graphs/contributors

Graph-Embeddings
################
Graph embeddings for downstream tasks

![graph_embeddings](https://raw.githubusercontent.com/rodrigo-arenas/graph-embeddings/main/docs/images/graph_embeddings.png)

Installation:
#############

It's advised to install graph-embeddings using a virtual env, inside the env use::

   pip install graph-embeddings

Algorithms:
###########

StackedNode2Vec
---------------

Computes the Node2Vec representation of each node in a set of graphs.

Example:

.. code-block:: python

   import networkx as nx
   from graph_embeddings.algorithms import StackedNode2Vec

   g1 = nx.DiGraph()
   g2 = nx.DiGraph()
   g1.add_edges_from([("A", "B"), ("B", "C"), ("C", "B"), ("B", "E")])
   g2.add_edges_from([("A", "B"), ("B", "D"), ("D", "C"), ("C", "D")])

   graphs = [g1, g2]
   embedding_model = StackedNode2Vec()
   embedding_model.fit(graphs)

   embedding_model.get_embeddings()  # ndarray with shape (5, 128, 2) - nodes, embedding_size, graphs
   embedding_model.get_dense_embeddings()  # ndarray with shape (2, 640) - graphs, nodes*embedding_size


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
the `Contribution guide <https://github.com/rodrigo-arenas/graph-embeddings/blob/main/CONTRIBUTING.md>`_.



