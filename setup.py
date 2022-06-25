import os
import pathlib
from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*

# get __version__ from _version.py
ver_file = os.path.join("graph_embeddings", "_version.py")
with open(ver_file) as f:
    exec(f.read())

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="graph-embeddings",
    version=__version__,
    description="Graph embeddings for downstream tasks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigo-arenas/Graph-Embeddings",
    author="Rodrigo Arenas",
    author_email="rodrigo.arenas456@gmail.com",
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",

    ],
    project_urls={
        "Documentation": "https://graph-embeddings.readthedocs.io/en/stable/",
        "Source Code": "https://github.com/rodrigo-arenas/Graph-Embeddings",
        "Bug Tracker": "https://github.com/rodrigo-arenas/Graph-Embeddings/issues",
    },
    packages=find_packages(include=['graph_embeddings', 'graph_embeddings.*']),
    install_requires=[
        'networkx>=2.8.4',
        'stellargraph>=1.2.1',
        'chardet>=5.0.0',
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
