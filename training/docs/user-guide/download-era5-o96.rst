####################
 ERA5 training data
####################

.. warning::

   Do not train a model using the URL below. You will need to download
   it locally first. The dataset is quite large (about 0.5 TB) and is
   composed of over 65,000 files.

ECMWF provides a dataset of ERA5 reanalysis data on a O96 `octahedral
reduced Gaussian grid
<https://confluence.ecmwf.int/display/FCST/Introducing+the+octahedral+reduced+Gaussian+grid>`__,
which has a resolution of approximately 1°.

The dataset contains data from the `Copernicus Climate Data Store
<https://cds.climate.copernicus.eu>`__ and is available under the
`Licence to use Copernicus Products
<https://object-store.os-api.cci2.ecmwf.int/cci2-prod-catalogue/licences/licence-to-use-copernicus-products/licence-to-use-copernicus-products_b4b9451f54cffa16ecef5c912c9cebd6979925a956e3fa677976e0cf198c2c18.pdf>`__.

The dataset can be download from
https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr.

*************************
 Downloading the dataset
*************************

To download the dataset, you can use the ``anemoi-dataset copy`` command
line tool available from the :ref:`anemoi-datasets
<anemoi-datasets:index-page>` package. You will need verion ``0.5.22``
or above.

.. code::

   % pip install "anemoi-datasets>=0.5.22"
   % anemoi-dataset copy \
       --url https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr \
       --target era5-o96-1979-2023-6h-v8.zarr

By default, the download will process 100 files at a time, in one
thread. If your internet connection is fast enough, you can increase the
number of threads using the ``--transfers`` option. If your internet
connection is slow, you can decrease the number files processed at a
time using the ``--blocks`` option.

If the download fails, you can resume the download using the
``--resume`` option, this will skip the blocks that have already been
downloaded.

.. note::

   The HTTP server hosting the dataset will limit the **overall** number
   of simultaneous connections. This means that your download may be
   affected by other users downloading the same data. If you get an
   error ``429 Too many requests``, simply restart the download with
   ``--resume``, and lower the number of threads.

************************
 Content of the dataset
************************

To do.
