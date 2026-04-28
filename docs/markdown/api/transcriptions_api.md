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

# The Transcriptions API

The transcription is the means by which Dymos converts the problem of finding a continuous control function which optimizes the operation of some system into a discrete optimization problem which can be solved by an off-the-shelf optimizer.

Dymos currently supports the following transcriptions

- Radau Pseudospectral Method
- Gauss-Lobatto Collocation
- Explicit Shooting

## Radau Options

```python
# tags: remove-input
import openmdao.api as om
import dymos as dm
tx = dm.transcriptions.Radau(num_segments=1, order=5)
om.show_options_table(tx)
```

## GaussLobatto Options

```python
# tags: remove-input
tx = dm.transcriptions.GaussLobatto(num_segments=1, order=5)
om.show_options_table(tx)
```

## Birkhoff Options

```python
tx = dm.transcriptions.Birkhoff(num_nodes=5, grid_type='cgl')
om.show_options_table(tx)
```

## ExplicitShooting Options

```python
# tags: remove-input
tx = dm.transcriptions.ExplicitShooting(num_segments=1, order=5)
om.show_options_table(tx)
```

## PicardShooting Options

```python
# tags: remove-input
tx = dm.transcriptions.PicardShooting(num_segments=1, nodes_per_seg=5)
om.show_options_table(tx)
```
