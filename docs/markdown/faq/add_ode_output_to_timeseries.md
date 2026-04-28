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

# How do I add an ODE output to the timeseries outputs?

The timeseries object in Dymos provides a transcription-independent way to get timeseries output of a variable in a phase.
By default, these timeseries outputs include Dymos phase variables (times, states, controls, and parameters).
Often, there will be some other intermediate or auxiliary output calculations in the ODE that we want to track over time.
These can be added to the timeseries outputs using the `add_timeseries_output` method on Phase.

Multiple timeseries outputs can be added at one time by matching a glob pattern.
For instance, to add all outputs of the ODE to the timeseries, one can use '*' as the `name` argument.

See the [Timeseries documentation](../features/phases/timeseries.ipynb) for more information.

The [commercial aircraft example](../examples/commercial_aircraft/commercial_aircraft.ipynb) uses the `add_timeseries_output` method to add the lift and drag coefficients to the timeseries outputs.
