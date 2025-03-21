.. _anemoi-models:

.. _index-page:

###########################################
 Welcome to `anemoi-models` documentation!
###########################################

.. warning::

   This documentation is work in progress.

The *anemoi-models* package is a collection of tools enabling you to
design custom models for training data-driven weather models. It is one
of the packages within the `anemoi framework
<https://anemoi-docs.readthedocs.io/en/latest/>`_.

**************
 About Anemoi
**************

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

****************
 Quick overview
****************

The *anemoi-models* package provides the core model components for used
by the rest of the *anemoi* packages to train graph neural networks for
data-driven weather forecasting.

-  :doc:`overview`
-  :doc:`installing`

************
 Installing
************

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-models

Get more information in the :ref:`installing <installing>` section.

**************
 Contributing
**************

.. code:: bash

   git clone https://github.com/ecmwf/anemoi-core.git
   cd anemoi-core/models
   pip install .[dev]

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc

***********************
 Other Anemoi packages
***********************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

..
   ..................................................................................

..
   From here defines the TOC in the sidebar, but is not rendered directly on the page.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   overview
   installing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   usage/create_model

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   modules/interface
   modules/models
   modules/layers
   modules/distributed
   modules/preprocessing
   modules/data_indices
