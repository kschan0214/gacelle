.. gacelle-documentation documentation master file, created by
   sphinx-quickstart on Tue Sep 24 13:55:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GACELLE Documentation
=====================

Welcome to the GACELLE documentation! Here you can find all the documents related to *GACELLE*.

*GACELLE* is a Matlab package that combines the flexibility of either conventional voxel-wise nonlinear-least-square(NLLS) optimization(askAdam) or Markov-chain-Monte-Carlo(MCMC) sampling with the high computational efficiency of GPU processing, for high throughput estimation tasks.

Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :name: sec-install

   installation/installation

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :name: sec-getstart

   getting_started/introduction
   getting_started/designing_model
   getting_started/askadam_basic_tutorial
   getting_started/askadam_basicND_tutorial
   getting_started/mcmc_basic_tutorial
   getting_started/mcmc_metropolishastings_tutorial
   getting_started/mcmc_affineinvariantensemble_tutorial

.. toctree::
   :maxdepth: 1
   :caption: Supported models
   :name: sec-supportedmodel

   supported_models/AxCaliberSMT
   supported_models/NEXI
   supported_models/MCRMWI
   supported_models/GREMWI
   supported_models/JointR1R2star

.. toctree::
   :maxdepth: 1
   :caption: Tips and Tricks
   :name: sec-tips

   tipsandtricks/general

.. toctree::
   :maxdepth: 1
   :caption: API
   :name: sec-api

   api/askadam/optimisation
   api/mcmc/optimisation
   api/mcmc/metropolis_hastings
   api/mcmc/goodman_weare

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
