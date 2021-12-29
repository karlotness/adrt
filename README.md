# Approximate Discrete Radon Transform

[![adrt on PyPI](https://img.shields.io/pypi/v/adrt)][pypi]
[![Documentation](https://readthedocs.org/projects/adrt/badge/?version=latest)][docs]
[![Tests](https://github.com/karlotness/adrt/workflows/Tests/badge.svg)][tests]

This library provides an implementation of an approximate discrete
Radon transform (ADRT) and its inverse, as a Python module operating
on NumPy arrays.

## References

This implementation is based on descriptions in several publications:
- Martin L. Brady, [A Fast Discrete Approximation Algorithm for the Radon Transform Related Databases][brady98], SIAM Journal on Computing, 27.
- William H. Press, [Discrete Radon transform has an exact, fast inverse and generalizes to operations other than sums along lines][press06], Proceedings of the National Academy of Sciences, 103.
- Donsub Rim, [Exact and fast inversion of the approximate discrete Radon transform from partial data][rim20], Applied Mathematics Letters, 102.

## License

The code in this repository is licensed under a 3-clause BSD license.
See [LICENSE.txt](LICENSE.txt) for the license text.

We also make available several pre-built binary copies of this
software. The binary build for Windows includes additional license
terms for runtime code included as part of the software. Review the
LICENSE.txt file in the binary build package for more information.

[pypi]: https://pypi.org/project/adrt/
[docs]: https://adrt.readthedocs.io/en/latest/
[tests]: https://github.com/karlotness/adrt/actions
[brady98]: https://doi.org/10.1137/S0097539793256673
[press06]: https://doi.org/10.1073/pnas.0609228103
[rim20]: https://doi.org/10.1016/j.aml.2019.106159
