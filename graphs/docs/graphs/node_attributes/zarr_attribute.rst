###################
 From Zarr dataset
###################

Zarr datasets are the standard format to define data nodes in
_anemoi-graphs_. The user can define node attributes based on a zarr
dataset variable. For example, the following recipe will define an
attribute `land_mask` based on the _lsm_ variable of the dataset.

.. literalinclude:: ../yaml/attributes_nonmissingzarr.yaml
   :language: yaml

In addition, if an user is using "cutout" operation to build their
dataset, it may be helpful to create a `cutout_mask` to track the
provenance of the resulting nodes. An example is shown below:

.. literalinclude:: ../yaml/attributes_cutout.yaml
   :language: yaml
