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

# Phase Options

## Phase.options
General options available to all dymos Phases.

```python
# tags: remove-input
import dymos
phase = dymos.Phase()

om.show_options_table(phase)
```

The transcription is an instance of one of the [transcriptions](./transcriptions_api) available in Dymos.


Other options listed below control the behavior of various aspects of Phases.
While most have corresponding phase methods for setting these values, users may now use the `set` method of OpenMDAO's OptionsDictionary to set the values of these options.

## Phase.timeseries_options

These options control the behavior of timeseries within a Phase.

```python
# tags: remove-input
om.show_options_table(phase.timeseries_options)
```

## Phase.refine_options
These options control grid refinement within each Phase.

```python
# tags: remove-input
om.show_options_table(phase.refine_options)
```

## Phase.simulate_options
These options control the behavior of phase explicit simulation.

```python
# tags: remove-input
om.show_options_table(phase.simulate_options)
```

# Phase Methods

## set_time_options

```{eval-rst}
    .. automethod:: dymos.Phase.set_time_options
        :noindex:
```

## set_time_val

```{eval-rst}
    .. automethod:: dymos.Phase.set_time_val
        :noindex:
```


## add_state
```{eval-rst}
    .. automethod:: dymos.Phase.add_state
        :noindex:
```

## set_state_options
```{eval-rst}
    .. automethod:: dymos.Phase.set_state_options
        :noindex:
```

## set_state_val
```{eval-rst}
    .. automethod:: dymos.Phase.set_state_val
        :noindex:
```

## add_control
```{eval-rst}
    .. automethod:: dymos.Phase.add_control
        :noindex:
```

## set_control_options
```{eval-rst}
    .. automethod:: dymos.Phase.set_control_options
        :noindex:
```

## set_control_val
```{eval-rst}
    .. automethod:: dymos.Phase.set_control_val
        :noindex:
```

## add_polynomial_control
```{eval-rst}
    .. automethod:: dymos.Phase.add_polynomial_control
        :noindex:
```

## set_polynomial_control_options
```{eval-rst}
    .. automethod:: dymos.Phase.set_polynomial_control_options
        :noindex:
```

## set_polynomial_control_val
```{eval-rst}
    .. automethod:: dymos.Phase.set_polynomial_control_val
        :noindex:
```

## add_parameter
```{eval-rst}
    .. automethod:: dymos.Phase.add_parameter
        :noindex:
```

## set_parameter_options
```{eval-rst}
    .. automethod:: dymos.Phase.set_parameter_options
        :noindex:
```

## set_parameter_val
```{eval-rst}
    .. automethod:: dymos.Phase.set_parameter_val
        :noindex:
```

## add_timeseries
```{eval-rst}
    .. automethod:: dymos.Phase.add_timeseries
        :noindex:
```

## add_calc_expr
```{eval-rst}
    .. automethod:: dymos.Phase.add_calc_expr
        :noindex:
```

## add_timeseries_output
```{eval-rst}
    .. automethod:: dymos.Phase.add_timeseries_output
        :noindex:
```

## add_boundary_constraint
```{eval-rst}
    .. automethod:: dymos.Phase.add_boundary_constraint
        :noindex:
```

## add_path_constraint
```{eval-rst}
    .. automethod:: dymos.Phase.add_path_constraint
        :noindex:
```

## simulate
```{eval-rst}
    .. automethod:: dymos.Phase.simulate
        :noindex:
```

## duplicate
```{eval-rst}
    .. automethod:: dymos.Phase.duplicate
        :noindex:
```

## set_refine_options
```{eval-rst}
    .. automethod:: dymos.Phase.set_refine_options
        :noindex:
```

## interp
```{eval-rst}
    .. automethod:: dymos.Phase.interp
        :noindex:
```
