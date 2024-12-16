###############
 HEALPix Nodes
###############

This method allows us to define nodes based on the Hierarchical Equal
Area isoLatitude Pixelation of a sphere (HEALPix). The resolution of the
HEALPix grid is defined by the `resolution` parameter, which corresponds
to the number of refinements of the sphere.

.. code:: yaml

   nodes:
     data:  # name of the nodes
       node_builder:
         _target_: anemoi.graphs.nodes.HEALPixNodes
         resolution: 3
       attributes: ...

For reference, the following table shows the number of nodes and
resolution for each resolution:

.. csv-table:: HEALPix refinements specifications
   :file: ./healpix.csv
   :header-rows: 1

.. warning::

   This class will require the `healpy
   <https://healpy.readthedocs.io/>`_ package to be installed. You can
   install it with `pip install healpy`.
