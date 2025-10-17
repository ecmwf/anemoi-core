#####################
 From anemoi dataset
#####################

Anemoi datasets are the standard format to define data nodes in
:ref:`anemoi-graphs <anemoi-graphs:index-page>`. The user can define
node attributes based on an Anemoi dataset variable. For example, the
following recipe will create a mask of valid (not missing) values in the
first timestep of the ``lsm`` variable and store it in the attribute
``land_mask``:

.. literalinclude:: ../yaml/attributes_nonmissingzarr.yaml
   :language: yaml

Alternatively, to create a mask of non-zero values in the first timestep
of the ``lsm`` variable, use the ``NonzeroAnemoiDatasetNodes`` class:

.. literalinclude:: ../yaml/attributes_nonzerozarr.yaml
   :language: yaml

In addition, if an user is using "cutout" operation to build their
dataset, it may be helpful to create a ``cutout_mask`` to track the
provenance of the resulting nodes. An example is shown below:

.. literalinclude:: ../yaml/attributes_cutout.yaml
   :language: yaml

Similarly, when using the "grids" operation to combine datasets, you can
create a ``grids_mask`` to track which nodes came from which dataset.
The mask will be ``True`` for nodes from the first dataset and ``False``
for nodes from the second dataset. Here's an example:

.. literalinclude:: ../yaml/attributes_grids.yaml
   :language: yaml
