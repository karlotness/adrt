API Reference
=============

This section provides a full API reference for this library. The
functionality is split into several submodules, providing routines
with varying functionality. Basic routines are provided in :mod:`adrt`
and useful utilities are in :mod:`adrt.utils`.

Many of the functions in this module have requirements for the shapes
and dtypes of their inputs. The core numerical routines support inputs
with :term:`numpy:dtype` either :obj:`numpy.float32` or
:obj:`numpy.float64` which are referred to as ":class:`numpy.ndarray`
of :class:`float`".

Our main routines often have requirements for the :term:`shapes
<numpy:shape>` of input arrays. Often, these functions also support an
optional batch dimension which makes it possible to process multiple
independent inputs at once without looping in Python.

Frequently, we will refer to an "ADRT output" or an "ADRT output of
size N" which is an array with a shape matching:

.. code-block:: text

   (batch?, 4, 2*N-1, N)

which specifies

#. an *optional* batch dimension
#. a dimension for the ADRT quadrants, of size exactly four
#. a dimension for ADRT offsets with size given by the formula,
   referencing the last dimension of size ``N``
#. a dimension for the angles of size ``N``, which must be a power of
   two

.. toctree::
   :maxdepth: 1
   :caption: Contents

   reference.basic
   reference.utils
   reference.core
