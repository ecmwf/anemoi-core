.. _checkpoint_troubleshooting:

#####################################
 Checkpoint Pipeline Troubleshooting
#####################################

This guide helps diagnose and resolve common issues with the checkpoint
pipeline system.

*******************
 Quick Diagnostics
*******************

Component Discovery
===================

Check what components are available:

.. code:: python

   from anemoi.training.checkpoint import ComponentCatalog

   print("Available sources:", ComponentCatalog.list_sources())
   print("Available loaders:", ComponentCatalog.list_loaders())
   print("Available modifiers:", ComponentCatalog.list_modifiers())

Checkpoint Inspection
=====================

Inspect a checkpoint without full loading:

.. code:: python

   from anemoi.training.checkpoint import get_checkpoint_metadata
   from pathlib import Path

   metadata = get_checkpoint_metadata(Path("model.ckpt"))
   print(f"File size: {metadata.get('file_size_mb', 0):.1f} MB")
   print(f"Parameters: {metadata.get('num_parameters', 'unknown')}")

*****************************
 Common Issues and Solutions
*****************************

Configuration Issues
====================

Unknown checkpoint source type
------------------------------

**Error Message:**

.. code:: text

   CheckpointConfigError: Unknown checkpoint source: 'my_source'

**Causes:**

-  Typo in source type name
-  Source module not implemented or imported
-  Missing dependencies for specific source types

**Solutions:**

#. Check available sources:

   .. code:: python

      from anemoi.training.checkpoint import ComponentCatalog
      print(ComponentCatalog.list_sources())

#. Use correct ``_target_`` path in configuration:

   .. code:: yaml

      checkpoint_pipeline:
        stages:
          - _target_: my_module.MySource
            path: /path/to/checkpoint.ckpt

Failed to instantiate stage
---------------------------

**Error Message:**

.. code:: text

   CheckpointConfigError: Failed to instantiate pipeline stage from configuration

**Causes:**

-  Invalid ``_target_`` path
-  Missing required parameters
-  Import errors in target module

**Solutions:**

#. Verify the target path is importable

#. Check all required parameters are provided

#. Test import manually:

   .. code:: python

      # Test if your stage can be imported
      from my_module import MyStage
      stage = MyStage(param="value")

Environment and Dependencies
============================

No checkpoint components discovered
-----------------------------------

**Warning Message:**

.. code:: text

   No sources components were discovered

**Causes:**

-  Module not yet implemented
-  Import errors in component modules
-  Missing dependencies

**Solutions:**

#. Check implementation status (some modules may not be implemented yet)

#. Try importing manually to check for errors:

   .. code:: python

      try:
          from my_module import MySource
          print("Module imported successfully")
      except ImportError as e:
          print(f"Module not available: {e}")

#. Enable debug logging to see import issues:

   .. code:: python

      import logging
      logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

Remote downloads not working
----------------------------

**Error Message:**

.. code:: text

   ImportError: aiohttp is required for remote checkpoint downloads

**Solution:**

Install the remote dependencies:

.. code:: bash

   pip install anemoi-training[remote]

File and Path Issues
====================

Checkpoint file does not exist
------------------------------

**Error Message:**

.. code:: text

   CheckpointNotFoundError: Checkpoint file not found at /path/to/checkpoint.ckpt

**Solutions:**

#. Verify file path:

   .. code:: python

      from pathlib import Path

      path = Path("/path/to/checkpoint.ckpt")
      print(f"Path exists: {path.exists()}")
      print(f"Is file: {path.is_file()}")
      print(f"Parent exists: {path.parent.exists()}")

#. Check permissions:

   .. code:: bash

      ls -la /path/to/checkpoint.ckpt

#. Use absolute paths in configuration:

   .. code:: yaml

      checkpoint_pipeline:
        stages:
          - _target_: my_module.LocalSource
            path: /absolute/path/to/checkpoint.ckpt

Loading and Compatibility Issues
================================

Shape mismatch during loading
-----------------------------

**Error Message:**

.. code:: text

   CheckpointIncompatibleError: Unexpected key(s) in state_dict

**Solutions:**

#. Use non-strict loading (when using a custom loader):

   .. code:: yaml

      stages:
        - _target_: my_module.WeightsOnlyLoader
          strict: false

#. Compare checkpoint and model keys:

   .. code:: python

      import torch
      from anemoi.training.checkpoint import compare_state_dicts

      checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")
      checkpoint_dict = checkpoint.get("state_dict", checkpoint)

      missing, unexpected, mismatches = compare_state_dicts(
          checkpoint_dict,
          model.state_dict()
      )

      print(f"Missing: {missing}")
      print(f"Unexpected: {unexpected}")
      print(f"Shape mismatches: {mismatches}")

Cannot extract state dict
-------------------------

**Error Message:**

.. code:: text

   CheckpointValidationError: Cannot find model state in checkpoint

**Causes:**

-  Non-standard checkpoint format
-  Checkpoint uses different key names

**Solutions:**

#. Inspect checkpoint structure:

   .. code:: python

      import torch

      checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")
      print("Available keys:", list(checkpoint.keys())[:10])

#. Use format detection:

   .. code:: python

      from anemoi.training.checkpoint.formats import (
          detect_checkpoint_format,
          extract_state_dict,
      )

      fmt = detect_checkpoint_format("checkpoint.ckpt")
      print(f"Detected format: {fmt}")

Network and Remote Source Issues
================================

Failed to download checkpoint
-----------------------------

**Error Message:**

.. code:: text

   CheckpointSourceError: Failed to download from https://...

**Solutions:**

#. Check network connectivity:

   .. code:: bash

      curl -I https://example.com/model.ckpt

#. Download manually first:

   .. code:: bash

      wget https://example.com/model.ckpt -O local_model.ckpt

   Then use a local source stage:

   .. code:: yaml

      checkpoint_pipeline:
        stages:
          - _target_: my_module.LocalSource
            path: ./local_model.ckpt

AWS S3 authentication failed
----------------------------

**Error Message:**

.. code:: text

   CheckpointSourceError: Access denied to s3://bucket/model.ckpt

**Solutions:**

#. Check AWS credentials:

   .. code:: bash

      aws configure list
      aws s3 ls s3://bucket/

#. Set environment variables:

   .. code:: bash

      export AWS_ACCESS_KEY_ID=your-key-id
      export AWS_SECRET_ACCESS_KEY=your-secret-key
      export AWS_DEFAULT_REGION=us-east-1

Memory and Performance Issues
=============================

Out of memory during loading
----------------------------

**Error Message:**

.. code:: text

   RuntimeError: CUDA out of memory

**Solutions:**

#. Load on CPU first:

   .. code:: python

      import torch

      checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")
      model.load_state_dict(checkpoint["state_dict"])
      model = model.cuda()

#. Use weights-only loading to skip optimizer state

#. Clear GPU memory:

   .. code:: python

      import torch

      torch.cuda.empty_cache()
      print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

********************
 Advanced Debugging
********************

Enable Debug Logging
====================

.. code:: python

   import logging

   # Enable detailed checkpoint pipeline logging
   logging.getLogger("anemoi.training.checkpoint").setLevel(logging.DEBUG)

   # Enable all debug logging
   logging.basicConfig(level=logging.DEBUG)

Pipeline Execution Tracking
===========================

After pipeline execution, check context metadata:

.. code:: python

   result = await pipeline.execute(context)

   print("Pipeline execution metadata:")
   for key, value in result.metadata.items():
       if key.startswith("stage_"):
           print(f"  {key}: {value}")

Manual Stage Testing
====================

Test individual stages in isolation:

.. code:: python

   from anemoi.training.checkpoint import CheckpointContext

   context = CheckpointContext(model=my_model)

   try:
       result = await my_stage.process(context)
       print("Stage succeeded")
   except Exception as e:
       print(f"Stage failed: {e}")
       import traceback
       traceback.print_exc()

**************
 Getting Help
**************

Report Issues
=============

When reporting issues, include:

#. Full error message and stack trace

#. Configuration file (sanitized)

#. Environment information:

   .. code:: python

      import sys
      import torch
      import anemoi.training

      print(f"Python: {sys.version}")
      print(f"PyTorch: {torch.__version__}")
      print(f"Anemoi Training: {anemoi.training.__version__}")
      print(f"CUDA available: {torch.cuda.is_available()}")

#. Component discovery output:

   .. code:: python

      from anemoi.training.checkpoint import ComponentCatalog

      print("Sources:", ComponentCatalog.list_sources())
      print("Loaders:", ComponentCatalog.list_loaders())
      print("Modifiers:", ComponentCatalog.list_modifiers())

Debug Information Collection
============================

.. code:: python

   def collect_debug_info():
       import sys, torch, platform

       info = {
           "python_version": sys.version,
           "pytorch_version": torch.__version__,
           "platform": platform.platform(),
           "cuda_available": torch.cuda.is_available(),
       }

       from anemoi.training.checkpoint import ComponentCatalog

       info["sources"] = ComponentCatalog.list_sources()
       info["loaders"] = ComponentCatalog.list_loaders()
       info["modifiers"] = ComponentCatalog.list_modifiers()

       return info

   import json
   print(json.dumps(collect_debug_info(), indent=2))
