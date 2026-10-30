#################
 Dataset Caching
#################

Anemoi can keep a copy of the training data on the local disk of each
node. This copy is the cache. The cache can decrease the load on the
shared filesystem.

**********
 Settings
**********

Use these settings in ``system.hardware``:

.. code:: yaml

   system:
     hardware:
       # Keep the cache under $TMPDIR.
       # To disable the cache, set cache_dir to null.
       cache_dir: ${oc.decode:${oc.env:TMPDIR}}
       # Write records in a background thread.
       # When a write is in progress, the next record is not written.
       cache_async_writes: false
       # Largest size of the cache on each node, as a fraction of the
       # total size of the disk.
       cache_max_fraction: 0.9
       # Text to add to each hostname for connections between nodes.
       # Use the hostname of the high-speed network interface,
       # for example "-ab-gpil-ib". Null uses the hostname.
       hostname_suffix: null

**************
 How it works
**************

The dataloader workers read each record (one date of one dataset) from
the cache if it is available. If the record is not in the cache, the
worker reads it from the filesystem. Then the worker writes it to the
cache.

-  Each node has one cache. All the GPUs on the node use the same cache.
-  The cache is deleted at the start of each run.
-  The records are compressed with Blosc (lz4).

When the training uses more than one node, a worker can read records
from the cache of a different node. Each node has a cache server for
this purpose. At the end of each epoch, the nodes share a list of the
records in each cache. Thus, the first epoch reads records only from the
local cache or from the filesystem.

*************
 Limitations
*************

-  The size of the cache on each node is limited to
   ``cache_max_fraction`` of the total size of the disk. All the
   processes on the node use the same limit. The cache also keeps
   ``1 - cache_max_fraction`` of the disk free. This limit includes
   other files on the disk, for example checkpoints. When the cache
   gets to a limit, or when the disk is full, the cache stops writing
   new records.
   After this, the cache does not change until the end of the run. The
   cache does not remove old records.

-  The cache gives a benefit only when the filesystem is slow or has a
   high load. On a node with no other load, a read from the cache can
   take the same time as a read from the filesystem.

-  If ``hostname_suffix`` is not correct for your system, the nodes
   cannot connect to the cache servers of other nodes. Then the workers
   read these records from the filesystem.
