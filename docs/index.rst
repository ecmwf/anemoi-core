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

   overview

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   start/installing
   start/training-your-first-model
   start/hydra-intro

.. toctree::
   :maxdepth: 1
   :caption: Using Anemoi Training

   user-guide/introduction
   user-guide/configuring
   user-guide/training
   user-guide/models
   user-guide/tracking
   user-guide/distributed
   user-guide/debugging

.. toctree::
   :maxdepth: 1
   :caption: Developing Anemoi Training

   dev/contributing
   dev/code-structure
   dev/hydra
   dev/testing

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Modules

   modules/*

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
