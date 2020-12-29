---
title: 'adrt: A Python package for the approximate discrete Radon transform'
tags:
  - Python
  - C++
  - numerical algorithms
  - fast transforms
  - image processing
  - Radon transform
  - approximate discrete Radon transform
authors:
  - name: Karl Otness
    orcid: 0000-0001-8534-2648
    affiliation: 1
  - name: Donsub Rim
    orcid: 0000-0002-6721-2070
    affiliation: 1
affiliations:
 - name: Courant Institute, New York University
   index: 1
date: 27 December 2020
bibliography: paper.bib
---

# Summary

The Radon transform is a fundamental integral transform that arises in many
different fields including medical/seismic tomography, signal/image processing,
and the analysis of partial differential equations. The `adrt` package 
implements a discretization of the Radon transform called the approximate
discrete Radon transform (ADRT) [@Brady1998]. The ADRT is a fast transform with
the complexity $\mathcal{O}(N^2 \log N)$ for a square of image of size $N
\times N$. The ADRT can approximate the Radon transform with the rate
$\mathcal{O} (N^{-1} \log N)$, and it possesses important inversion properties
[@Press2006; @Rim2020] enabling it to be used to approximate the inverse of the
Radon transform.

`adrt` implements the core functions in `C++` with `OpenMP` parallelism which
can be imported in Python via a wrapper. These include the fast forward
transform, the back-projection, the single-quadrant inverse. It also provides
python utilities including an iterative inverse using the `scipy` conjugate
gradient implementation [@scipy] and an interpolation routine to output to a
uniform grid.

# Statement of need

`adrt` aims to facilitate numerical experimentation with the ADRT by providing
an implementation of basic operations of the ADRT algorithm. This will be
useful in two broad aspects. There are various open research problems regarding
the ADRT itself, and this package will be useful for researchers studying
the properties of the ADRT. On the other hand, the ADRT was found to be useful
in scientific computing applications [@Rim2018] and is potentially useful in
various applications in imaging, image processing and machine learning. `adrt`
will provide an accessible implementation for these applications as well.

# Related research and software

Various other discretizations and approximations of the Radon transform exists.
A linear interpolation and filtered back-projection is implemented in
[@scikit-image]; the discrete Radon transform [@Beylkin87]; a fast transform
based on the pseudo-polar Fourier transform [@ACDISS2008]; the non-uniform fast
Fourier transform (NUFFT) [@GreengardLee2004; @BMK2019]. However the ADRT has
unique properties that distinguishes it from other discretizations, such as the
localization property and the range characterization [@LRR2020], its locality
and simplicity.

# Acknowledgements

This work was partially supported by by the Air Force Center of Excellence on
Multi-Fidelity Modeling of Rocket Combustor Dynamics under Award Number
FA9550-17-1-0195 and AFOSR MURI on multi-information sources of multi-physics
systems under Award Number FA9550-15-1-0038.


# References
