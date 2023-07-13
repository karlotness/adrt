# Approximate Discrete Radon Transform

[![adrt on PyPI](https://img.shields.io/pypi/v/adrt)][pypi]
[![adrt on conda-forge](https://img.shields.io/conda/vn/conda-forge/adrt.svg)][condaforge]
[![Documentation](https://readthedocs.org/projects/adrt/badge/?version=latest)][docs]
[![Tests](https://github.com/karlotness/adrt/actions/workflows/test.yml/badge.svg)][tests]
[![JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.05083/status.svg)][joss]

Fast approximate discrete Radon transform for
[NumPy](https://numpy.org/) arrays.

- **Documentation:** https://adrt.readthedocs.io/en/latest/
- **Source Code:** https://github.com/karlotness/adrt
- **Bug Reports:** https://github.com/karlotness/adrt/issues

This library provides an implementation of an approximate discrete
Radon transform (ADRT) and related routines as a Python module
operating on NumPy arrays. Implemented routines include: the forward
ADRT, a back-projection operation, and several inverse transforms. The
package [documentation][docs] contains usage examples, and sample
applications.

## Installation

Install from [PyPI][pypi] using pip:
``` console
$ python -m pip install adrt
```
or from [conda-forge][condaforge]:
``` console
$ conda install -c conda-forge adrt
```

For further details on installation or building from source, consult
the [documentation][docs].

## Citation

If you use this software in your research, please cite our associated
[JOSS paper][joss].

```BibTeX
@article{adrt,
  title={adrt: approximate discrete {R}adon transform for {P}ython},
  author={Karl Otness and Donsub Rim},
  journal={Journal of Open Source Software},
  publisher={The Open Journal},
  year=2023,
  doi={10.21105/joss.05083},
  url={https://doi.org/10.21105/joss.05083},
  volume=8,
  number=83,
  pages=5083,
}
```

## References

This implementation is based on descriptions in several publications:
- Martin L. Brady, [A Fast Discrete Approximation Algorithm for the Radon Transform Related Databases][brady98], SIAM Journal on Computing, 27.
- William H. Press, [Discrete Radon transform has an exact, fast inverse and generalizes to operations other than sums along lines][press06], Proceedings of the National Academy of Sciences, 103.
- Donsub Rim, [Exact and fast inversion of the approximate discrete Radon transform from partial data][rim20], Applied Mathematics Letters, 102.

## License

This software is distributed under the 3-clause BSD license. See
LICENSE.txt for the license text.

We also make available several pre-built binary copies of this
software. The binary build for Windows includes additional license
terms for runtime code included as part of the software. Review the
LICENSE.txt file in the binary build package for more information.

[pypi]: https://pypi.org/project/adrt/
[condaforge]: https://anaconda.org/conda-forge/adrt
[docs]: https://adrt.readthedocs.io/en/latest/
[tests]: https://github.com/karlotness/adrt/actions
[joss]: https://doi.org/10.21105/joss.05083
[brady98]: https://doi.org/10.1137/S0097539793256673
[press06]: https://doi.org/10.1073/pnas.0609228103
[rim20]: https://doi.org/10.1016/j.aml.2019.106159
