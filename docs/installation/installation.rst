.. _installation:
.. role::  raw-html(raw)
    :format: html

Installation
======================

The latest update can be found on `GitHub <https://github.com/kschan0214/gacelle>`_.

You need to add the directory of the package to your Matlab's PATH

``addpath(genpath('/path/to/gacelle/'))``

It is recommended to use the latest Matlab version for the best compatibility of their Deep Learning Toolbox. We did most of the development and testings on R2023a, though in principle the package should compatibile to earlier Matlab versions back to R2019b (not tested).

You also need an NVIDIA GPU to be able to use GPU computing in Matlab. Check `this page <https://www.mathworks.com/help/parallel-computing/gpu-computing-requirements.html>`_ for more information about GPU computing in Matlab.
