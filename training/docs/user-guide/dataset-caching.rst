#################
 Dataset Caching
#################

Anemoi can keep a copy of the training data on the local disk of each
node. This copy is the cache. The cache can decrease the load on the
shared filesystem.

**********
 Settings
**********

The cache is off by default (``cache_dir: null``). To turn it on, set
``cache_dir`` to a directory on a local disk of each node, for example
``$TMPDIR``. Use these settings in ``system.hardware``:

.. code:: yaml

   system:
     hardware:
       # Keep the cache under $TMPDIR.
       # The default is null: no cache.
       cache_dir: ${oc.decode:${oc.env:TMPDIR}}
       # The cache stops writing when less than
       # (1 - cache_max_fraction) of the disk is free.
       cache_max_fraction: 0.9
       # Text to add to each hostname for connections between nodes.
       # Use the hostname of the high-speed network interface,
       # for example "-ib". Null uses the hostname.
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
-  The records are compressed with Blosc (lz4). A record holds only the
   variables and grid points that the training uses.

When the training uses more than one node, the dates of each dataset are
divided into one contiguous part for each node. Each node caches only the
records of its part. Each node has a cache server. When a record is not
in the local cache, the worker asks the node of its part for the record.
If that node does not have the record, the worker reads it from the
filesystem. Thus, the nodes together can keep a dataset that is too
large for the disk of one node.

A node caches a record of its part only when one of its ranks reads that
record. Thus, in the first epochs, some records are not in any cache, and
the workers read them from the filesystem.

With model sharding, a record is one grid shard of one date. Each node
caches all the grid shards that its ranks read, because the other nodes
possibly do not read the same shards. A worker does not ask other nodes
for grid shards.

*************
 Limitations
*************

-  The cache keeps ``1 - cache_max_fraction`` of the disk free. This
   limit includes other files on the disk, for example checkpoints.
   When less space is free, or when the disk is full, the cache stops
   writing new records.
   After this, the cache does not change until the end of the run. The
   cache does not remove old records.

-  The cache gives a benefit only when a read from the filesystem
   costs more than a training step, and a read from the cache costs
   less. Examples:

   -  The training uses only some variables of the dataset (``select``
      or ``drop``), or only one part of the grid (model sharding). Each
      zarr chunk holds all the variables and all the grid points. Thus,
      without the cache, each read gets and decompresses the full
      chunk. The cache keeps only the part that the training uses.
   -  The filesystem is slow or has a high load.

   If a read from the filesystem is fast, or if the training step is
   longer than the read, the cache gives no benefit. The first epoch
   writes the cache and takes approximately the same time as a run
   with no cache.

-  The cache works only when the dataloader workers start with
   ``fork``. This is the default on Linux. Do not set
   ``dataloader.multiprocessing_context`` to ``spawn`` or
   ``forkserver`` when the cache is on. From Python 3.14, the default on
   Linux is ``forkserver``: then set it to ``fork``.

-  At the start of the run, each rank makes sure that it can find the
   address of each other node (the hostname and ``hostname_suffix``).
   If it cannot find an address, the run stops with an error. Then make
   ``hostname_suffix`` correct for your system, or set it to null.
