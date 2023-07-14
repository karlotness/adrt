Maintenance
===========

The sections below list some required steps for maintaining this
package, in particular, the process for releasing a new version, and
for modifying the version ranges for dependencies.

Release
-------

The release process is largely automated; the main steps are:

#. Update ``__version__`` in ``__init__.py`` and commit
#. Add a Git tag for the new version
#. Push the new commit, then the tag to GitHub
#. Wait for the release workflow to run
#. Check diagnostics (see below)
#. Authorize the upload to PyPI

First, make sure that the most recent commit to the main branch passed
all its tests.

Next, update the version string in ``__init__.py`` then make a commit
checking in that change and push it to GitHub. After that, tag that
commit with a "v" prefix, so for a version of ``1.2.3`` the tag is
``v1.2.3``. Push that tag to GitHub.

The release workflow will run tests and build several copies of the
package. The "draft-release" job produces information on the built
artifacts: their checksums, contents, and diagnostics for the native
extensions. These are displayed on the summary page for the overall
release workflow run.

Check the archive contents for stray files. The sdist ending in
``.tar.gz`` should include our Python source code and Python tests (no
C++ tests). The wheels ending in ``*.whl`` should include the Python
module source code, and one native module ``_adrt_cdefs``. There
should be no tests and no C++ sources in the wheels. Make sure no
third party libraries inadvertently got picked up and included in the
wheels.

For the diagnostics, the main thing to check is our symbol imports and
exports. Our native extension should export only one symbol
``PyInit__adrt_cdefs``, nothing else. For imports, we want to only
load C libraries at runtime (no ``libstdc++`` on Linux or ``MSVCP`` on
Windows, note the *P* suffix).

On Linux we should only link to ``libc``, ``libm``, and possibly
``libpthread`` (a spurious dependency).

On Windows we expect ``python3.dll``, ``KERNEL32.dll``, and any number
of ``api-ms-win-crt-*.dll``, and currently ``VCRUNTIME140.dll``. This
VC runtime version is decided by the current MSVC version and may have
a higher number in the future. ``VCRUNTIME140.dll`` is included with
all supported versions of Python. If the number has changed for a new
toolkit make sure the oldest version of Python that we target includes
the new DLL. Otherwise it may have to be bundled in the wheels.

On macOS we should only need ``libsystem``.

If all tests have passed, and there are no obvious warnings or errors,
authorize the final upload job in GitHub Actions which will upload the
built packages to PyPI. Finally, add a few release notes to the
`release <https://github.com/karlotness/adrt/releases>`__ that is
created automatically.

NumPy Versions
--------------

On occasion we should raise the minimum version of NumPy we accept.
There are generally two versions involved:

* the version of NumPy used to *build* the native extension using its
  C API
* the version of NumPy used at runtime

See below for more information on how these interact and should be
selected.

The version used for building should be chosen to be at least the
version required for runtime, and also more recent than NumPy version
1.25. The minimum build-time version should be set as a constraint in
``build-system.requires`` in ``pyproject.toml`` and as the value of
``NPY_NO_DEPRECATED_API`` in ``adrt_cdefs_py.cpp``.

The minimum runtime version should be older than the build time
version and this should be set as a constraint in
``project.dependencies`` in ``pyproject.toml`` and as the value for
``NPY_TARGET_VERSION`` in ``adrt_cdefs_py.cpp``.

Ensure that the minimum required versions for both build-time and
runtime support the oldest version of Python supported by the package
and that NumPy binary wheels are available for all platforms for which
we build wheels.

.. note::

   For both NumPy and Python versions we roughly follow `SPEC 0
   <https://scientific-python.org/specs/spec-0000/>`__ and `NEP 29
   <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

Python Versions
---------------

In general we want to support all actively maintained Python releases.
When a version of Python goes out of support we may raise our lower
version limit, and when a new version is released we will want to add
it to our tests.

In either case there are three places to update the version:

* GitHub Actions workflows (the ``setup-python`` steps in ``test.yml``
  and ``release.yml``)
* ``pyproject.toml`` (``project.requires_python`` and
  ``tool.cibuildwheel.test-skip``)
* ``adrt_cdefs_py.cpp`` (``Py_LIMITED_API``)

In general when raising the minimum version, replace locations using
the old minimum with the new minimum version. Similarly, when adding a
new Python release, replace the previous ceiling with the new one and
shift versions around (as may be needed in the GitHub Actions
workflows).

The ``Py_LIMITED_API`` macro should always use the current minimum
version and should match ``project.requires_python``.

Be mindful of test and build dependencies. In particular, SciPy adds
wheels later than NumPy so new and prerelease Python versions may need
to have tests skipped both by *temporarily* leaving off Python
versions GitHub Actions ``setup-python`` steps and by blocking their
tests in ``tool.cibuildwheel.test-skip``.

On Windows, NumPy and SciPy are gradually phasing out 32-bit builds.
For Python 3.10 and later there are no SciPy wheels for 32-bit
windows. Block all 32-bit Windows tests on Python 3.10 and later and
once we only support these versions, 32-bit builds can be dropped
altogether.

.. note::

   A new Python release *does not* necessarily require a new release
   of this package. So long as CPython supports `abi3
   <https://docs.python.org/3/c-api/stable.html>`__ our old builds
   should work on new Python versions.

Linter Versions
---------------

We restrict versions of some linters to avoid surprise errors from
upgrades or inconsistencies between local runs and checks on GitHub
Actions. These should be updated to the most recent version
periodically. Check PyPI for the most recent version and update the
pins under ``[testenv:lint]`` in ``tox.ini``.

In particular, we use `Black <https://pypi.org/project/black/>`__ to
style our Python source code. This should be limited to a range from
the latest version through the current year.

After updating any linter versions, run the linters locally and fix
any new style issues.
