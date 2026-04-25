.. _cutoff_radius:

################
 Cut-off radius
################

The *cut-off method* is a method for establishing connections between
two sets of nodes. Given two sets of nodes, (`source`, `target`), the
cut-off method connects all source nodes, :math:`V_{source}`, in a
neighbourhood of the target nodes, :math:`V_{target}`.

.. image:: ../../_static/cutoff.jpg
   :alt: Cut-off radius image
   :align: center

The neighbourhood is defined by a `cut-off radius`, which can be
specified in two mutually exclusive ways:

#. **Using cutoff_factor**: The radius is computed as the product of a
   factor and the reference distance:

.. math::

   \text{cutoff_radius} = \text{cutoff_factor} \times \text{nodes_reference_dist}

where :math:`\text{nodes_reference_dist}` is the maximum distance
between a target node and its nearest source node.

.. math::

   \text{nodes_reference_dist} = \max_{x \in V_{target}} \left\{  \min_{y \in V_{source}, y \neq x} \left\{ d(x, y) \right\} \right\}

where :math:`d(x, y)` is the `Haversine distance
<https://en.wikipedia.org/wiki/Haversine_formula>`_ between nodes
:math:`x` and :math:`y`. The :math:`\text{cutoff_factor}` is a parameter
that can be adjusted to increase or decrease the size of the
neighbourhood, and consequently the number of connections in the graph.

2. **Using cutoff_distance_km**: The radius can be directly specified in
   kilometers, which is converted to the appropriate unit sphere
   distance.

To use this method to create your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
      # Using cutoff_factor (relative to grid reference distance)
      -  source_name: source
         target_name: destination
         edge_builders:
         -  _target_: anemoi.graphs.edges.CutOffEdges
            cutoff_factor: 0.6
            # max_num_neighbours: 64

      # Or using cutoff_distance_km (direct distance specification)
      -  source_name: source
         target_name: destination
         edge_builders:
         -  _target_: anemoi.graphs.edges.CutOffEdges
            cutoff_distance_km: 300.0
            # max_num_neighbours: 64

The optional argument ``max_num_neighbours`` (default: 64) can be set to
limit the maximum number of neighbours each node can connect to.

.. note::

   The cut-off method is recommended for the encoder edge, to connect
   all data nodes to hidden nodes. The optimal ``cutoff_factor`` value
   will be the lowest value without isolated nodes. This optimal value
   depends on the node distribution, so it is recommended to tune it for
   each case.

#########################
 Reversed Cut-off radius
#########################

The reversed cut-off method (``ReversedCutOffEdges``) is similar to the
standard cut-off method, but instead establishes connections based on
the neighbourhood of each source node. The role of source and target
nodes is similarly reversed in :math:`\text{nodes_reference_dist}`.
Given two sets of nodes, (`source`, `target`), the
``ReversedCutOffEdges`` method connects all sources nodes to all target
nodes within a cut-off radius.

To use this method to create your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
      -  source_name: source
         target_name: destination
         edge_builders:
         -  _target_: anemoi.graphs.edges.ReversedCutOffEdges
            cutoff_factor: 0.6
