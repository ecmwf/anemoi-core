.. _anemoi-graphs:

.. _index-page:

###########################################
 Welcome to `anemoi-graphs` documentation!
###########################################

.. warning::

   This documentation is work in progress.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

The `anemoi-graphs` package allows you to design custom graphs for
training data-driven weather models. The graphs are built using a
`recipe`, which is a YAML file that specifies the nodes and edges of the
graph.

-  :doc:`overview`

.. toctree::
   :maxdepth: 1
   :hidden:

   overview

*****************
 Building graphs
*****************

-  :doc:`graphs/introduction`
-  :doc:`graphs/node_coordinates`
-  :doc:`graphs/node_attributes`
-  :doc:`graphs/edges`
-  :doc:`graphs/edge_attributes`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Building graphs

   graphs/introduction
   graphs/node_coordinates
   graphs/node_attributes
   graphs/edges
   graphs/edge_attributes

*********
 Modules
*********

-  :doc:`modules/node_builder`
-  :doc:`modules/edge_builder`
-  :doc:`modules/node_attributes`
-  :doc:`modules/edge_attributes`
-  :doc:`modules/graph_creator`
-  :doc:`modules/graph_inspector`
-  :doc:`modules/post_processor`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Modules

   modules/node_builder
   modules/edge_builder
   modules/node_attributes
   modules/edge_attributes
   modules/graph_creator
   modules/graph_inspector
   modules/post_processor

*******************
 Command line tool
*******************

-  :doc:`cli/introduction`
-  :doc:`cli/create`
-  :doc:`cli/describe`
-  :doc:`cli/inspect`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Command line tool

   cli/introduction
   cli/create
   cli/describe
   cli/inspect

**************************
 Developing Anemoi Graphs
**************************

-  :doc:`dev/contributing`
-  :doc:`dev/code_structure`
-  :doc:`dev/testing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developing Anemoi Graphs

   dev/contributing
   dev/code_structure
   dev/testing

***********
 Tutorials
***********

-  :doc:`usage/getting_started`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Usage

   usage/getting_started
   usage/limited_area

*****************
 Anemoi packages
*****************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html
