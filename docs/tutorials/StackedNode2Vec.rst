StackedNode2Vec

Understanding StackedNode2Vec
=============================

Introduction
------------

StackedNode2Vec is a method for finding a vector representation for each node in a graph; it has two steps:
A Random Walk to generate "phrases" for each node of the Graph and a Word2Vec algorithm to find
the vector representation of the phrases, i.e., the embeddings of each node.

In this API, the Node2Vec of each node is called the "embedding", combining all the nodes' embeddings of one graph
to create a single embedding is called a "dense embedding"; for the StackedNode2Vec method, a simple stack (flatten)
of all the individual embeddings is performed, returning a matrix where each row vector
is the dense embedding of each graph.

With the StackedNode2Vec API you can control the parameters of the random walk and from the Node2Vec algorithm,
check the API's documentation for more information.