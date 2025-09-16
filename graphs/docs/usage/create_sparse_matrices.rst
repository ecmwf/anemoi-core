.. _usage-create_sparse_matrices:

###########################################
 Create sparse matrices with anemoi-graphs
###########################################

Anemoi graphs supports the creation of sparse matrices using the
``export_to_sparse`` command. See the :ref:`Export To Sparse Command
<export-to-sparse-command>` for details.

The main use case in Anemoi is to create the truncation matrices used in
the models for the residual connection as explained :ref:`here
<anemoi-training:usage-field_truncation>`.

**************************
 Create your graph recipe
**************************

You can define your graph in a YAML configuration file, following the
same approach described in the other usage sections. This YAML file
should specify the nodes, edges, and any attributes required for your
graph. For more details on how to structure your recipe, refer to the
examples in the :ref:`usage-getting_started` and
:ref:`usage-limited_area` sections.

.. literalinclude:: yaml/sparse_matrices.yaml
   :language: yaml

*****************************************
 Export the sparse matrices to NPZ files
*****************************************

To export the sparse matrices, use the ``anemoi-graphs
export_to_sparse`` command as shown below. This command will process
your graph recipe and generate one sparse matrix for each edge attribute
in each subgraph of your graph.

Each sparse matrix is saved as an independent NPZ file in the specified
output directory. This allows for easy loading and use of the matrices
in downstream applications.

For example, running the following command:

.. code:: bash

   % anemoi-graphs export_to_sparse graph_recipe.yaml truncation_matrices/

will create a set of NPZ files in the ``truncation_matrices/``
directory, with each file corresponding to a specific edge attribute in
a subgraph.

***********************************************
 Select specific edges or attributes to export
***********************************************

By default, the ``export_to_sparse`` command will export sparse matrices
for all edge attributes in all subgraphs defined in your graph recipe.
However, you can restrict the export to only a subset of edge attributes
or subgraphs using the optional arguments ``--edges-attributes-name``
and ``--edges-name``.

-  ``--edges-attributes-name``: Only export sparse matrices for the
   specified edge attribute(s). You can provide this argument multiple
   times to select several attributes.

-  ``--edges-name``: Only export sparse matrices for the specified
   subgraph(s) (i.e., specific edge sets as defined in your recipe). You
   can provide this argument multiple times to select several subgraphs.

For example, to export only the ``gauss_weight`` attribute for the
``data->down`` subgraph, you can run:

.. code:: bash

   % anemoi-graphs export_to_sparse graph_recipe.yaml output_dir/ \
       --edges-attributes-name gauss_weight \
       --edges-name data->down

You can specify multiple attributes or subgraphs by repeating the
arguments:

.. code:: bash

   % anemoi-graphs export_to_sparse graph_recipe.yaml output_dir/ \
       --edges-attributes-name gauss_weight \
       --edges-attributes-name another_weight \
       --edges-name data->down \
       --edges-name down->data

This flexibility allows you to generate only the sparse matrices you
need for your application, reducing storage and processing time.
