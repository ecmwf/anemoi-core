.. _graphs-introduction:

##############
 Introduction
##############

In `anemoi-graphs`, graphs are built using a `recipe`, which is a YAML
file that specifies the nodes, edges, and attributes of an
encoder-processor-decoder graph. The recipe file is used to create the
graphs using the :ref:`command-line tool <cli-introduction>`.

.. image:: ../_static/enc_proc_dec.png
   :alt: Encoder-processor-decoder graph
   :align: center

The main components of the recipe file are the definition of the
`nodes`, the `edges`, and each of these can optionally include
`attributes`. Here we give an overview of these specifications; more
details on the available options for each are given in the following
pages.

*******
 Nodes
*******

The `nodes` are defined in the ``nodes`` section of the recipe file. The
keys are the names of the sets of `nodes` that will later be used to
build the connections. Each `nodes` configuration must include a
``node_builder`` section describing how to define the `nodes`, and can
optionally include node `attributes`.

A simple example of the nodes definition, for two sets of nodes, looks
like this:

.. literalinclude:: ../usage/yaml/nodes.yaml
   :language: yaml

In this example, two sets of nodes are defined, which have been named
``data`` and ``hidden`` (these names are later used to define edges).
The ``data`` nodes have been defined based on coordinates specified in a
:ref:`zarr file <zarr-file>` and the hidden nodes are specified based on
a :ref:`triangular refined icosahedron <trinodes>` algorithm.

Several methods are currently supported to create your nodes. You can
use indistinctly any of these to create your `data` or `hidden` nodes:

-  :doc:`node_coordinates/zarr_dataset`
-  :doc:`node_coordinates/npz_file`
-  :doc:`node_coordinates/reduced_gaussian`
-  :doc:`node_coordinates/icon_mesh`
-  :doc:`node_coordinates/text_file`
-  :doc:`node_coordinates/latlon_arrays`
-  :doc:`node_coordinates/tri_refined_icosahedron`
-  :doc:`node_coordinates/hex_refined_icosahedron`
-  :doc:`node_coordinates/healpix`

In addition to the ``node_builder`` section, the `nodes` configuration
can contain an optional section to define additional node attributes
(weights, mask, ...). For example, the weights can be used to define the
importance of each node in the loss function, or the masks can be used
to build connections only between subsets of nodes. See the
:ref:`attributes <graphs-node_attributes>` page for more details.

*******
 Edges
*******

The ``edges`` section in the recipe file defines the edges between the
nodes through which information will flow. These connections are defined
between pairs of `nodes` sets (source and target, specified by
`source_name` and `target_name`). There are several methods to build
these edges, including cutoff (`CutOffEdges`) or nearest neighbours
(`KNNEdges`).

For an encoder-processor-decoder graph you will need to build two sets
of `edges`. The first set of edges will connect the `data` nodes with
the `hidden` nodes to encode the input data into the latent space,
normally referred to as the `encoder edges` and represented here by the
first element of the ``edges`` section. The second set of `edges` will
connect the `hidden` nodes with the `data` nodes to decode the latent
space into the output data, normally referred to as `decoder edges` and
represented here by the second element of the ``edges`` section.

Graphically, the encoder-processor-decoder setup looks like this:

.. figure:: ../usage/schemas/global_wo-proc.png
   :alt: Schema of global graph (without processor connections)
   :align: center
   :width: 250

The corresponding recipe file chunk is as follows:

.. literalinclude:: ../usage/yaml/global_wo-proc.yaml
   :language: yaml

In this example, the encoder edges are defined based on the
:ref:`cut-off radius <cutoff_radius>`, whereas the decoder nodes use a
:ref:`k-nearest neighbours <knn>` algorithm. Available methods for
defining edges are:

-  :ref:`Cut-off radius <cutoff_radius>`
-  :ref:`K-nearest neighbours <knn>`
-  :ref:`Multi-scale connections <multi_scale>`

As with the nodes, the edges can have additional attributes - see the
:ref:`attributes <edge-attributes>` page for more details.
