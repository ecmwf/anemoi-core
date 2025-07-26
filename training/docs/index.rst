.. _anemoi-training:

.. _index-page:

#############################################
 Welcome to `anemoi-training` documentation!
#############################################

.. warning::

   This documentation is work in progress.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

This package provides the *Anemoi* training functionality.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/overview
   introduction/installing

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user-guide/overview
   user-guide/basic-set-up
   user-guide/hydra-intro
   user-guide/configuring
   user-guide/kcrps-set-up
   user-guide/diffusion-set-up
   user-guide/training
   user-guide/models
   user-guide/tracking
   user-guide/benchmarking
   user-guide/distributed
   user-guide/download-era5-o96

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: API Reference

   modules/*

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing

.. toctree::
   :maxdepth: 1
   :caption: FAQ/Troubleshooting

   troubleshooting

*****************
 Anemoi packages
*****************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html
