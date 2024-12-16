######
 Data
######

This module is used to initialise the dataset (constructed using
anemoi-datasets) and load in the data in to the model. It also performs
a series of checks, for example, that the training dataset end date is
before the start date of the validation dataset.

``dataset.py`` contains functions which define how the dataset gets
split between the workers (``worker_init_func``) and how the dataset is
iterated across to produce the data batches that get fed as input into
the model (``__iter__``).

Users wishing to change the format of the batch input into the model
should sub-class ``NativeGridDataset`` and change the ``__iter__``
function.

.. automodule:: anemoi.training.data.dataset
   :members:
   :no-undoc-members:
   :show-inheritance:
