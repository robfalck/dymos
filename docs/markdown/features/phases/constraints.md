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

# Constraints

Now that we've shown how to add degrees of freedom to a system with variables in the form of
time, states, and controls, we need to look at how to constrain the system.  In optimal control,
constraints typically come in two flavors:  boundary constraints and path constraints.

As OpenMDAO components, outputs of Dymos Phases can be constrained using OpenMDAO's `add_constraint` method, but Dymos Phases provide their own methods to make defining these constraints somewhat simpler.

##  Boundary Constraints

Boundary constraints are constraints on a variable value at the start or end of a phase.  There
are a few different ways to impose these in Dymos, each with slightly different behavior.

Let's consider that we want to solve for the elevation angle that results in the maximum possible
range flown by a cannonball.  In this situation we have some set of initial conditions that are
fixed.

\begin{align}
    t_0 &= 0 \, \mathrm{s} \\
    x_0 &= 0 \, \mathrm{m} \\
    y_0 &= 0 \, \mathrm{m} \\
    v_0 &= 100 \, \frac{\mathrm{m}}{\mathrm{s}}
\end{align}

The first, most obvious way to constrain fixed values is to remove them from the optimization problem altogether.

For time, this is done using the `fix_initial` or `fix_duration` arguments to `set_time_options`.
This also allows `t_initial` and `t_duration` to be provided from an external source via connection, if so desired.

For states and controls, the situation is slightly different.
Rather than providing initial and final values, similar to the way time is handled, the implicit simulation techniques must
be provided state values at the state discretization nodes and control values at *all* nodes.
Instead, for states and controls, the user specifies `fix_initial=True` or `fix_final=True`.

Removing constrained values from the optimization has the following pros and cons.
On the pro side, we're making the optimization problem smaller by omitting them.
On the con side, the optimizer has absolutely no freedom to move these values around even a little.
This can sometimes lead to failure modes that aren't necessarily obvious, especially to new users.

The following example solves the brachistochrone problem by omitting the initial time and initial state, as well as the final position state from the optimization.

The second method for bounding initial/final time, states, or controls is to leave them in the
optimization problem but to constrain only their initial or final values.  For time, this is
accomplished with the options `initial_bounds` and `duration_bounds`.  Each of these takes a tuple
of `(lower, upper)` values that the optimizer must obey when providing new variable values.  Note
that since states and controls may be vector valued, lower and upper may themselves be iterable.
To *pin* the value of a state, time, or control to a value just set lower and upper to the same
value.

As for the pros and cons of this technique, its largely similar to that for the first technique,
but it's somewhat optimizer dependent.  Some optimizers *may* allow bounds on design variables to
be violated slightly (to some small tolerance).  In theory this could alleviate some of the issues
with omitting a design variable altogether, but in practice that's unlikely.

The first two options work by imposing bounds (or by not providing a variable to the optimizer
altogether).  The third option is to pose bound constraints as actual constraints on the NLP.
This is accomplished with the `add_boundary_constraint` method on Phases.

The downside of this technique is that it makes the NLP problem larger, though not by much.  On
the plus side, this method allows the user to constrain any output within the ODE.  If the user
needs to constrain an auxiliary output, this is the only option.  It may also behave somewhat better
in certain circumstances.  Depending on scaling, the NLP may ensure that collocation defects are
satisfied before forcing an infeasible boundary constraint to be satisfied, for instance.

In conclusion, while using `fix_initial=True` for problems with fixed initial conditions is not a bad solution, the generality of `add_boundary_constraint`, especially for terminal constraints that risk being over-constrained, makes it a good first-choice in those situations.
In forward-shooting phases (`solve_segments='forward'`) only the initial values of the states are design variables for the optimizer.
As such, simple bounds on final state values are not possible in those situations, and `add_boundary_constraint` must be used instead.

##  Path Constraints

The second class of constraints supported by Dymos are *path* constraints, so called because they are imposed throughout the entire phase.
Like bound constraints, path constraints can be imposed on design variables using simple bounds.
This is accomplished using the `lower` and `upper` arguments to `add_state`, `add_control`, and `add_parameter`.
(Since time is monotonically increasing or decreasing the notion of a path constraint is irrelevant for it).

For vector-valued states and controls, lower/upper should be dimensioned the same as state or control.
If given as a scalar, it will be applied to all values in the state or control.

```{Note}
Bounds on states in Gauss-Lobatto Phases are **not** equivalent to path constraints.
The values of states in Gauss-Lobatto phases are provided at only the state-transcription nodes and then interpolated to the collocation nodes.
Therefore, the bounds will have no impact on these interpolated values which therefore may not satisfy the bounds, as one might expect.
``` 

Phases also support the `add_path_constraint` method, which imposes path constraints as constraints in the NLP problem.
As with `add_boundary_constraint`, the `add_path_constraint` method is the only option for path constraining an output of the ODE.

The downside of path constraints is that they add a considerable number of constraints to the NLP problem and thus may negatively impact performance, although this is generally minor for many problems.

##  Constraining Expressions

Constraints may be defined using mathematical expressions of the form `y=f(x)` to be evaluated. Here `x` may be vector combination of time, states, controls, parameters, or any outputs of the ODE. The variable `y` is added to the timeseries outputs of the phase and the desired constraint is applied to it.

Consider, again, the example of maximizing the range flown by a cannonball. But now, rather than a constraint on the initial velocity, we wish to apply a constraint to the initial normalized kinetic energy.

\begin{align}
    ke &= 0.5 * v^2 \\
    ke_0 &= 5000 \, \frac{\mathrm{m^2}}{\mathrm{s^2}}
\end{align}

The first way to achieve this is to add kinetic energy as a state in the model. This state may then be constrained either using `fix_initial=True` or `add_boundary_constraint(‘ke’, loc=’initial’ , equals=5000)`.

The second method to add this constraint is to add the constraint using an expression. This may be done as `add_boundary_constraint(‘ke=0.5*v**2, loc=’initial’, equals=5000)`.

The advantage to the latter method is that no changes are required to the previous model, simply modifying the `add_boundary_constraint` statement is sufficient. The disadvantage is that the derivatives of the constraints are evaluated using complex step rather than analytical expressions. This may negatively impact performance, but the effect should be minor for most problems.

## Implementation Detail - Constraint Aliasing

As of OpenMDAO Version 3.17.0, multiple constraints are allowed to exist on the same output as long as they use different indices and are provided different aliases.
In Dymos, this allows us to always apply multiple constraints (initial, final, or path constraints) to the timeseries outputs of the phase.
To allow boundary and path constraints to potentially be applied to the same timeseries outputs, they are provided the following aliases:

An initial boundary constraint on the name `'alpha'` will be given the alias `f'{path_to_phase}.initial_boundary_constraints->alpha`.  

A final boundary constraint on the name `'alpha'` will be given the alias `f'{path_to_phase}.final_boundary_constraints->alpha`.

A path constraint on the name `'alpha'` will be given the alias `f'{path_to_phase}.path_constraints->alpha`.  

The use of the `->` in this case is intended to remind the user that this is not the actual path to the variable being constrained.

##  Constraint Linearity

OpenMDAO will treat all boundary and path consraints as nonlinear unless the user provides the argument `linear=True` to `add_boundary_constraint` or `add_path_constraint`.
Note that it is the responsibility of the user to understand the inner workings of Dymos and their model well enough to know if the constraint may be treated as linear.
Specifying an output that is actually a nonlinear function of the design variables as a linear constraint will almost certainly result of the failure of the optimization due to incorrect derivatives.
The derivatives of linear constraints are computed once and cached by OpenMDAO for the remainder of the optimization.

