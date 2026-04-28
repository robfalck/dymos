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

# The Trajectory API

## add_linkage_constraint
```{eval-rst}
    .. automethod:: dymos.Trajectory.add_linkage_constraint
        :noindex:
```

## add_parameter
```{eval-rst}
    .. automethod:: dymos.Trajectory.add_parameter
        :noindex:
```

## set_parameter_val
```{eval-rst}
    .. automethod:: dymos.Trajectory.set_parameter_val
        :noindex:
```

## add_phase
```{eval-rst}
    .. automethod:: dymos.Trajectory.add_phase
        :noindex:
```

## link_phases
```{eval-rst}
    .. automethod:: dymos.Trajectory.link_phases
        :noindex:
```

## simulate
```{eval-rst}
    .. automethod:: dymos.Trajectory.simulate
        :noindex:
```
