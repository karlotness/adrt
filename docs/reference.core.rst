Core Routines
=============

.. automodule:: adrt.core

Generator Routines
------------------

The main algorithms implemented in this package are internally
iterative. For example, the ADRT merges line segments, doubling their
lengths until they span the full size of the input.

The low-level routines here make it possible to observe the progress
of these iterations. These are :term:`python:generator` functions that
will yield snapshots of the iterative computation after each step.
Their final snapshots are equivalent to the outputs of the basic
algorithms in the :mod:`adrt` module.

.. autofunction:: adrt_iter

.. autofunction:: bdrt_iter

.. autofunction:: iadrt_fmg_iter

Single-Step Routines
--------------------

These routines allow selecting a particular iteration to run on an
input. Internally, the iterative routines select the number of
iterations required based on the size of an input using the
:func:`num_iters` function. Values in this range are then valid as
loop counter values to be passed to the ``step`` arguments of these
functions.

The outputs of these routines allow modifying the progress of the
algorithm, either by editing the results of the steps (for example, to
mask certain values) or by running the steps multiple times or in a
different order.

.. autofunction:: num_iters

.. autofunction:: adrt_init

.. autofunction:: adrt_step

.. autofunction:: bdrt_step

.. autofunction:: iadrt_fmg_step

Multithreading Status
---------------------

Most of the core routines in this package can be built with support
for OpenMP. The function :func:`threading_enabled` queries whether
this support is enabled.

.. autofunction:: threading_enabled
