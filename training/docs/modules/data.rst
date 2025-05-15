######
 Data
######

This module is used to initialise the dataset (constructed using
anemoi-datasets) and load in the data in to the model. It also performs
a series of checks, for example, that the training dataset end date is
before the start date of the validation dataset.

The dataset files contain functions which define how the dataset gets
split between the workers (``worker_init_func``) and how the dataset is
iterated across to produce the data batches that get fed as input into
the model (``__iter__``).

.. note::

   Users wishing to change the format of the batch input into the model
   should sub-class ``NativeGridDataset`` and change the ``__iter__``
   function.

The ``singledataset.py`` file contains the ``NativeGridDataset`` class
which is used for deterministic model training.

.. automodule:: anemoi.training.data.dataset.singledataset
   :members:
   :no-undoc-members:
   :show-inheritance:

The ``ensdataset.py`` file contains the ``EnsNativeGridDataset`` class
which is used for ensemble model training.

.. automodule:: anemoi.training.data.dataset.ensdataset
   :members:
   :no-undoc-members:
   :show-inheritance:
