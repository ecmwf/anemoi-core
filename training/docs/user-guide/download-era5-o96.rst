.. _download-era5-o96:

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
which has a resolution of approximately 1°. The dataset provides
6-hourly data for the period from 1979 to 2023. The list of variable is
provided below.

The dataset contains data from the `Copernicus Climate Data Store
<https://cds.climate.copernicus.eu>`__ and is available under the
`CC-BY-4.0 license <https://creativecommons.org/licenses/by/4.0/>`__.

The dataset can be download from
https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr.

*************************
 Downloading the dataset
*************************

To download the dataset, you can use the `anemoi-datasets` :ref:`copy
<anemoi-datasets:copy_command>` command. You will need version
``0.5.22`` of the package or above.

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

.. list-table:: Pressure level variables
   :header-rows: 1

   -  -  Variable
      -  Description
      -  Units

   -  -  q
      -  Specific humidity
      -  kg/kg

   -  -  t
      -  Temperature
      -  K

   -  -  u
      -  U-component of wind
      -  m/s

   -  -  v
      -  V-component of wind
      -  m/s

   -  -  w
      -  Vertical velocity
      -  Pa/s

   -  -  z
      -  Geopotential
      -  m²/s²

Each of the variables above are named in the dataset as
``<variable>_<level>``. For example, the Geopotential at 1000hPa is name
``z_1000``. The pressure levels are: 1000, 925, 850, 700, 600, 500, 400,
300, 250, 200, 150, 100 and 50 hPa.

.. list-table:: Single level variables
   :header-rows: 1

   -  -  Variable
      -  Description
      -  Units

   -  -  10u
      -  U-component of wind at 10m
      -  m/s

   -  -  10v
      -  V-component of wind at 10m
      -  m/s

   -  -  2d
      -  Dew point temperature at 2m
      -  K

   -  -  2t
      -  Air temperature at 2m
      -  K

   -  -  cp
      -  Convection precipitation (6h accumulation)
      -  m

   -  -  lsm
      -  Land-sea mask
      -  0-1

   -  -  msl
      -  Mean sea level pressure
      -  Pa

   -  -  sdor
      -  Standard deviation of sub-gridscale orography
      -  m

   -  -  skt
      -  Skin temperature
      -  K

   -  -  slor
      -  Slope of sub-gridscale orography
      -  .

   -  -  sp
      -  Surface pressure
      -  Pa

   -  -  tcw
      -  Total column water
      -  m

   -  -  tp
      -  Total precipitation (6h accumulation)
      -  m

   -  -  z
      -  Orography
      -  m²/s²

.. list-table:: Forcing variables
   :header-rows: 1

   -  -  Variable
      -  Description
      -  Units

   -  -  cos_latitude
      -  Cosine of latitude
      -  .

   -  -  cos_longitude
      -  Cosine of longitude
      -  .

   -  -  sin_latitude
      -  Sine of latitude
      -  .

   -  -  sin_longitude
      -  Sine of longitude
      -  .

   -  -  cos_julian_day
      -  Cosine of Julian day
      -  .

   -  -  cos_local_time
      -  Cosine of local time
      -  .

   -  -  sin_julian_day
      -  Sine of Julian day
      -  .

   -  -  sin_local_time
      -  Sine of local time
      -  .

   -  -  insolation
      -  Insolation
      -  .

For more information on the forcing variables, see the :ref:`forcings
<anemoi-datasets:forcing_variables>` in the `anemoi-datasets` package.
