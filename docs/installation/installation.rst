.. _installation:
.. role::  raw-html(raw)
    :format: html

Installation
======================

The latest update can be found on `GitHub <https://github.com/kschan0214/gacelle>`_.

You need to add the directory of the package to your Matlab's PATH. As of
v1.1, the recommended way is to call the bundled ``addpath_gacelle``
function, which adds GACELLE and its subfolders to the path while
excluding ``docs/``, ``sandbox/``, ``deprecated/`` and ``mpl_training/``
(unlike a plain ``genpath``, which would add those too):

.. code-block::

    addpath('/path/to/gacelle/');
    addpath_gacelle();

If you need the raw, unfiltered behaviour instead, the old approach still
works:

``addpath(genpath('/path/to/gacelle/'))``

It is recommended to use the latest Matlab version for the best compatibility of their Deep Learning Toolbox. We did most of the development and testings on R2023a, though in principle the package should be compatible to R2022b.

You also need an NVIDIA GPU to be able to use GPU computing in Matlab. Check `this page <https://www.mathworks.com/help/parallel-computing/gpu-computing-requirements.html>`_ for more information about GPU computing in Matlab.
