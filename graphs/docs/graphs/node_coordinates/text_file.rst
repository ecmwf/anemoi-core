################
 From text file
################

To define the `node coordinates` based on a `.txt` file, you can
configure the `.yaml` as follows:

.. code:: yaml

   nodes:
     data: # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.TextNodes
         dataset: my_file.txt
         idx_lon: 0
         idx_lat: 1

Here, dataset refers to the path of the `.txt` file that contains the
latitude and longitude values in the columns specified by `idx_lat` and
`idx_lon`, respectively.
