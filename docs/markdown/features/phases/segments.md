```python
# tags: active-ipynb, remove-input, remove-output
# This cell is mandatory in all Dymos documentation notebooks.
missing_packages = []
try:
    import openmdao.api as om  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !python -m pip install openmdao[notebooks]
    else:
        missing_packages.append('openmdao')
try:
    import dymos as dm  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !python -m pip install dymos
    else:
        missing_packages.append('dymos')
try:
    import pyoptsparse  # noqa: F401
except ImportError:
    if 'google.colab' in str(get_ipython()):
        !pip install -q condacolab
        import condacolab
        condacolab.install_miniconda()
        !conda install -c conda-forge pyoptsparse
    else:
        missing_packages.append('pyoptsparse')
if missing_packages:
    raise EnvironmentError('This notebook requires the following packages '
                           'please install them and restart this notebook\'s runtime: {",".join(missing_packages)}')
```

# Segments

All phases in Dymos are decomposed into one or more *segments* in time.  These segments
serve the following purposes:

- Gauss-Lobatto collocation and the Radau Pseudospectral method model each state variable as a polynomial segment in nondimensional time within each segment.
- Each control is modeled as a polynomial in nondimensional time within each segment.

The order of the *state* polynomial segment is given by the phase argument `transcription_order`.
In Dymos the minimum supported transcription order is 3.

State-time histories within a segment are modelled as a Lagrange polynomial.
Continuity in state value may be enforced via linear constraints at the segment boundaries (the default behavior) or by specifying a *compressed* transcription whereby the state value at a segment boundary is provided as a single value.
The default *compressed* transcription yields an optimization problem with fewer variables, but in some situations using uncompressed transcription can result in more robust convergence.
