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
       # Largest size of the cache on each node, as a fraction of the
       # total size of the disk.
       cache_max_fraction: 0.9
       # Divide the dates of each dataset into one part for each node.
       # Each node caches only the records of its part (default).
       cache_partition: true
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
-  The records are compressed with Blosc (lz4). If each chunk of the
   source zarr is one date with Blosc compression, the cache writes the
   chunk as it is. It does not compress the record again.

When the training uses more than one node, a worker can read records
from the cache of a different node. Each node has a cache server for
this purpose. At the end of each epoch, the nodes share a list of the
records in each cache. Thus, the first epoch reads records only from the
local cache or from the filesystem.

By default (``cache_partition: true``), the dates of each dataset are
divided into one contiguous part for each node. Each node caches only
the records of its part, and reads the other records from the other
nodes. Thus, the nodes together can keep a dataset that is too large
for the disk of one node. The partition applies only to records of the
full grid. With model sharding, each node caches all the grid shards
that its ranks read, because the other nodes possibly do not read the
same shards.

With ``cache_partition: false``, each node caches all the records that
its ranks read. Each date is in more than one sample, for example in 3
samples when the model uses 2 input dates and 1 output date. Thus, each
node caches most of the dates: with 2 nodes, approximately 87% of the
dates are in the cache of each node.

With the partition, more records come from other nodes. In a test on 2
nodes, a read from another node took the same time as a local read. But
the second and third epochs have more misses: a node does not keep a
record that is not in its part. If the node of that part did not read
the record in the first epoch, no node has it yet.

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

-  At the start of the run, each rank makes sure that it can find the
   address of each other node (the hostname and ``hostname_suffix``).
   If it cannot find an address, the run stops with an error. Then make
   ``hostname_suffix`` correct for your system, or set it to null.
