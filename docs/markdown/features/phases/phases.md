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

# Phases of a Trajectory

Dymos uses the concept of *phases* to support intermediate boundary constraints and path constraints on variables in the system.
Each phase represents the trajectory of a dynamical system, and may be subject to different equations of motion, force models, and constraints.
Multiple phases may be assembled to form one or more trajectories by enforcing compatibility constraints between them.

For implicit and explicit phases, the equations-of-motion or process equations are defined via an ordinary differential equation.

An ODE is of the form

\begin{align}  
    \frac{\partial \textbf x}{\partial t} = \textbf f(t, \textbf x, \textbf u)
\end{align}

where
$\textbf x$ is the vector of *state variables* (the variable being integrated),
$t$ is *time* (or *time-like*),
$\textbf u$ is the vector of *parameters* (an input to the ODE),
and
$\textbf f$ is the *ODE function*.

Dymos can treat the parameters $\textbf u$ as either static **parameters** or dynamic **controls**.
In addition, Dymos automatically calculates the first and second time-derivatives of the controls.
These derivatives can then be utilized as via constraints or as additional parameters to the ODE.
Subsequently, the optimal control problem as solved by Dymos can be expressed as:

\begin{align}
  \textrm{Minimize}:& \quad J = \textbf f_{obj}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \\
  \textrm{subject to:}& \\
  &\textrm{system dynamics} \quad &\frac{\partial \textbf x}{\partial t} &= \textbf f_{ode}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \\
  &\textrm{initial time bounds} \quad &t_{0,lb} &\,\le\, t_0 \,\le\, t_{0,ub} \\
  &\textrm{elapsed time bounds} \quad &t_{p,lb} &\,\le\, t_p \,\le\, t_{p,ub} \\
  &\textrm{state bounds} \quad &\textbf x_{lb} &\,\le\, \textbf x \,\le\, \textbf x_{ub} \\
  &\textrm{control bounds} \quad &\textbf u_{lb} &\,\le\, \textbf u \,\le\, \textbf u_{ub} \\
  &\textrm{nonlinear boundary constraints} \quad &\textbf g_{b,lb} &\,\le\, \textbf g_{b}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \,\le\, \textbf g_{b,ub} \\
  &\textrm{nonlinear path constraints} \quad &\textbf g_{p,lb} &\,\le\, \textbf g_{p}(t, \textbf x, \textbf u, \dot{\textbf u}, \ddot{\textbf u}) \,\le\, \textbf g_{p,ub} \\
\end{align}

The ability to utilize control derivatives in the equations of motion provides some unique capabilities, namely the ability to
easily solve problems using _differential inclusion_, which will be demonstrated in the examples.

The solution techniques used by the Phase classes in Dymos generally fall into three categories:
implicit, explicit, and analytic.  They differ in underlying details but mostly allow for the same
general form of the optimal control problem.

Implicit solution techniques (using the _Radau Pseudospectral Method_ or _Gauss-Lobatto Collocation_ solve the ODE "all at once" and iteratively change the values of the state variables until the rates of change from the dynamics agree with the rates of change obtained by fitting polynomials to the states over time.
Typically, some parameters and control variables affect the path of the state variables.

Explicit solution techniques are typical _shooting_ techniques that propagate the dynamics forward in time from some given initial condition.
Like the implicit techniques, using an explicit numerical integration allows for some set of parameters or controls to affect the dynamics of the system.

With an analytic solution, there is no numeric integration occuring because the solution is known analytically. These phases are a fast way to propagate simple dynamics when an analytic solution is avaiable.
The downside to analytic solutions is that they don't allow for the input of a general time-varying control.
Such systems are only controlled static _parameters_ of the phase.

## Features of Implicit and Explicit Phases

[Analytic Phases](analytic_phases.ipynb)

[Segments](segments.ipynb)

[Variables](variables.ipynb)

[Constraints](constraints.ipynb)

[Objective](objective.ipynb)

[Timeseries Outputs](timeseries.ipynb)

[Calculation Expressions](calc_expressions.ipynb)


