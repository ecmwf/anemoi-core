#################
 Troubleshooting
#################

When encountering issues while training models with Anemoi Training,
this guide will help you identify and resolve common problems. We'll
cover various debugging techniques, including those specific to PyTorch
Lightning, which Anemoi Training uses under the hood.

****************************
 Using Debug Configurations
****************************

To troubleshoot errors when trying to train a model for the first time,
it is advisable to use the debug configuration
``anemoi/training/config/debug.yaml``. This configuration:

-  Runs a small model
-  Trains on a limited number of batches per epoch
-  Helps identify errors more quickly

If you're using a custom configuration, consider making these temporary
adjustments:

.. code:: yaml

   dataloader:
     limit_batches:
       training: 100
       validation: 100

   hardware:
     num_gpus_per_node: 1

These settings limit the data processed and use a single GPU, helping
isolate issues related to data or parallelization.

****************
 PyTorch errors
****************

1. Error when opening checkpoints > 2 GB
========================================

If you're using a version of PyTorch < 2.6 with checkpoint files > 2 GB
you may encounter an error such as

.. code:: python

   RuntimeError: PytorchStreamReader failed reading zip archive: invalid header or archive is corrupted

This is due to a limitation with the zip implementation. If you're
unable to move to a more recent version of PyTorch you can try to remove
the metadata stored in the checkpoint using

.. code:: bash

   anemoi-utils metadata checkpoint.ckpt --remove --output output.ckpt

***********************************
 PyTorch Lightning Debugging Tools
***********************************

Anemoi Training leverages PyTorch Lightning, which provides several
useful debugging tools.

Currently these aren't implemented as config settings yet, but could
easily be added, if needed.

1. Overfit on a Single Batch
============================

To identify issues in your model's ability to learn, try overfitting on
a single batch:

.. code:: python

   # use only 1% of the train & val set
   trainer = Trainer(overfit_batches=0.01)

   # overfit on 10 of the same batches
   trainer = Trainer(overfit_batches=10)

This setting will repeatedly train on the same batch, helping you verify
if the model can learn at all.

2. Fast Dev Run
===============

For a quick test of your entire training pipeline:

.. code:: python

   trainer = Trainer(fast_dev_run=True)

This runs a single batch for training, validation, and testing, checking
if all code paths work without errors.

3. Detect Anomalies
===================

Enable PyTorch's anomaly detection in the diagnostics configuration:

.. code:: yaml

   debug:
       anomaly_detection: true

This helps identify issues like NaN or infinity values in your model's
computations.

***************************************
 Debug Flags for Better Error Handling
***************************************

Anemoi Training can make use of several debug flags to provide more
detailed error information:

1. Verbose Mode
===============

Enable verbose logging:

.. code:: yaml

   hydra.verbose=true

You can set the log level of the logger NAME to DEBUG. Equivalent to
``import logging; logging.getLogger(NAME).setLevel(logging.DEBUG)``.

.. code:: yaml

   hydra.verbose=NAME

And even provide multiple targets.

.. code:: yaml

   hydra.verbose=[NAME1,NAME2]

This increases the verbosity of log outputs, providing more detailed
information about the training process.

2. Asynchronous Callbacks
=========================

Disable asynchronous callbacks for clearer error messages:

.. code:: yaml

   diagnostics:
     plot:
       asynchronous: false

This makes error messages generally easier to understand by ensuring
callbacks are executed synchronously.

3. Disable Plotting
===================

Turn off plotting callbacks to isolate non-visualization related issues:

.. code:: yaml

   diagnostics:
     plot:
       callbacks: []

**********************************
 Debugging C10 Distributed Errors
**********************************

The C10 distributed error can often mask underlying issues. To debug the
true model error:

1. Set CUDA to Blocking Mode
============================

Before running your training script, set the following environment
variable:

.. code:: bash

   export CUDA_LAUNCH_BLOCKING=1

This forces CUDA operations to run synchronously, which can reveal the
true source of errors that might be hidden by asynchronous execution.

2. Run on a Single GPU
======================

Temporarily run your model on a single GPU to eliminate some distributed
training complexities:

.. code:: yaml

   hardware:
     num_gpus_per_node: 1

The code is still distributed, but at least it removes the multi-GPU
aspect and you can use debug statements.

3. Gradually Increase Complexity
================================

Once you've identified and fixed the underlying issue, gradually
reintroduce distributed training and multiple GPUs to ensure the problem
doesn't reoccur in a multi-GPU setting.

*********************************
 Additional Troubleshooting Tips
*********************************

1. Check Input Data
===================

Verify that your input data is correctly formatted and addressed in the
normalizer. Use small subsets of your data to test the pipeline.

2. Inspect Model Outputs
========================

Regularly print or log model outputs, especially in the early stages of
training, to catch any anomalies.

3. Monitor Resource Usage
=========================

Keep an eye on CPU, GPU, and memory usage. Unexpected spikes or constant
high usage might indicate inefficiencies or leaks.

This can be enabled in the diagnostics configuration:

.. code:: yaml

   log:
       mlflow:
           system: true

4. Gradient Checking
====================

If you suspect issues with backpropagation, consider implementing
gradient checking to verify correct gradient computations.

****************************
 Seeking Further Assistance
****************************

If you've tried these troubleshooting steps and still encounter issues,
consider:

-  Reviewing the Anemoi Training documentation for any recent updates or
   known issues
-  Checking the project's issue tracker for similar problems and
   solutions
-  Reaching out to the Anemoi community or support channels for
   additional help

Remember to provide as much relevant information as possible when
seeking assistance, including your configuration, error messages, and
steps to reproduce the issue.
