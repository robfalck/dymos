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

# The run_problem function

In the Brachistochrone example we used two methods on the OpenMDAO Problem class to execute the model.

`run_model` Takes the current model design variables and runs a single execution of the Problem's model.  
Any iterative systems are converged, but no optimization is performed.
When using dymos with an optimizer-driven implicit transcription, `run_model` will **not** produce a physically valid trajectory on output.
If using a solver-driven transcription, the collocation defects will be satisfied (if possible) and the resulting outputs will provide a physically valid trajectory (to the extent possible given the collocation grid).

`run_driver` runs a driver wrapped around the model (typically done for optimization) and repeatedly executes `run_model` until the associated optimization problem is satisfied.
This approach will provide a physically valid trajectory, to the extent that the grid is sufficient to accurately model the dynamics.

But commonly, we want to do the following when we run dymos

- Automatically record states, controls, and parameters at the arrived-upon "final" solution.
- Automatically provide explicit simulation of the solution to verify the accuracy of the collocation.
- Automatically load in the results from a previous case as the initial guess for an optimization.
- Iteratively re-optimize the problem with different grid settings to attempt to minimize grid error (i.e. grid refinement).

To remove the need to repeatedly setup the code to do this, dymos provides a function called `run_problem`.

```{eval-rst}
    .. autofunction:: dymos.run_problem
        :noindex:
```
