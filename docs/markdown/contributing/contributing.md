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

# Contributing to Dymos

Dymos is open-source software and the developers welcome collaboration with the community on finding and fixing bugs or requesting and implementing new features.

## Found a bug in Dymos?

If you believe you've found a bug in Dymos, [submit a new issue](https://github.com/OpenMDAO/dymos/issues).
If at all possible, please include a functional code example which demonstrates the issue (the expected behavior vs. the actual behavior).

## Fixed a bug in Dymos?

If you believe you have a fix for an existing bug in Dymos, please submit the fix as [pull request](https://github.com/OpenMDAO/dymos/pulls).
Under the "related issues" section of the pull request template, include the issue resolved by the pull request using Github's [referencing syntax](https://help.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue).
When submitting a bug-fix pull request, please include a [unit test](https://docs.python.org/3.8/library/unittest.html) that demonstrates the corrected behavior.
This will prevent regressions in the future.

## Need new functionality in Dymos?

If you would like to have new functionality that currently doesn't exist in Dymos, please submit your request via [the Dymos issues on Github](https://github.com/OpenMDAO/dymos/issues).
The Dymos development team is small and we can't promise that we'll add every requested capability, but we'll happily have a discussion and try to accommodate reasonable requests that fit within the goals of the library.

## Adding new examples

Adding a new example is a great way to contribute to Dymos.
It's a great introduction to the Dymos development process, and examples provide a great way for users to learn to apply Dymos in new applications.
Submit new examples via [the Dymos issues on Github](https://github.com/OpenMDAO/dymos/issues).
A new example should do the following:

- Include a new directory under the `dymos/examples` directory.
- A unittest should be included in a `doc` subfolder within the example directory.
- The unittest method should be self-contained (it should include all imports necessary to run the example).
- If you want to include output and/or plots from the example in the documentation (highly recommended), decorate the test with the `@dymos.utils.doc_utils.save_for_docs` decorator.  This will save the text and plot outputs from the test for inclusion in the Dymos documentation.
- A new markdown file should be added under `mkdocs/docs/examples/<example name>` within the Dymos repository.

The Dymos docs are built on [JupyterBook](https://jupyterbook.org/intro.html) which allows users to run any page of the documentation by opening it in colab as a [Jupyter Notebook](https://jupyter.org). For those wanting to contribute, they are able to contribute by writing their own Jupyter Notebooks. Below are some important ways on how to build notebooks for Dymos.

## Notebook Creation

**Header**

At the begining of every notebook, we require (without exception) to have the following code cell at the top of every notebook with the three tags: `active-ipynb`, `remove-input`, `remove-output`. Tags can be added at the top of the notebook menu by going to `View` -> `Cell Toolbar` -> `Tags`. 

```python
# tags: active-ipynb, remove-input, remove-output
try:
    import openmdao.api as om  # noqa: F401
    import dymos as dm  # noqa: F401
except ModuleNotFoundError:
    !python -m pip install openmdao[notebooks]
    !python -m pip install dymos
    import openmdao.api as om  # noqa: F401
    import dymos as dm  # noqa: F401
```

**Adding Code Examples**

If you want to add a block of code, for example, simply add it to a code block like we have below.

```python
# tags: remove-input, hide-output
om.display_source("dymos.examples.brachistochrone.doc.brachistochrone_ode")
```

```python
import numpy as np
import openmdao.api as om
from dymos.examples.brachistochrone.doc.brachistochrone_ode import BrachistochroneODE

num_nodes = 5

p = om.Problem(model=om.Group())

ivc = p.model.add_subsystem('vars', om.IndepVarComp())
ivc.add_output('v', shape=(num_nodes,), units='m/s')
ivc.add_output('theta', shape=(num_nodes,), units='deg')

p.model.add_subsystem('ode', BrachistochroneODE(num_nodes=num_nodes))

p.model.connect('vars.v', 'ode.v')
p.model.connect('vars.theta', 'ode.theta')

p.setup(force_alloc_complex=True)

p.set_val('vars.v', 10*np.random.random(num_nodes))
p.set_val('vars.theta', 10*np.random.uniform(1, 179, num_nodes))

p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)
```

There should be a unit test associated with the code and it needs to be below the test. To keep the docs clean for users, we require that all tests be hidden (with few exceptions) using the tags `remove-input` and `remove-output`.

- On the off chance you want to show the assert, use the tag `allow_assert`. 
- If your output is unusually long, use the tag `output_scroll` to make the output scrollable.

Below is an assert test of the code above.

```python
# tags: allow-assert
from dymos.utils.testing_utils import assert_check_partials

assert_check_partials(cpd)
```

**Showing Source Code**

If you want to show the source code of a particular class, there is a utility function from OpenMDAO to help you. Use `om.display_source()` to display your code. Example below:

```{Note}
This should include the tag `remove-input` to keep the docs clean
```

```python
om.display_source("dymos.examples.brachistochrone.brachistochrone_ode")
```

**Citing**

If you want to cite a journal, article, book, etc, simply add ```{cite}`youbibtextname` ``` next to what you want to cite. Add your citiation to `reference.bib` so that keyword will be picked up by JupyterBook. Below is an example of a Bibtex citation, that citation applied, and then a reference section with a filter to compile a list of the references mentioned in this notebook.

```
@inproceedings{gray2010openmdao,
  title={OpenMDAO: An open source framework for multidisciplinary analysis and optimization},
  author={Gray, Justin and Moore, Kenneth and Naylor, Bret},
  booktitle={13th AIAA/ISSMO Multidisciplinary Analysis Optimization Conference},
  pages={9101},
  year={2010}
}
```

Grey {cite}`gray2010openmdao`

### References

```{bibliography}
:filter: docname in docnames
```

**Building Docs**

When you want to build the docs, run the following line from the top level of the Dymos folder: `jupyter-book build docs/`

## Running Tests

Dymos tests can be run with any test runner such as [nosetests](https://nose.readthedocs.io/en/latest/) or [pytest](https://docs.pytest.org/en/stable/).
However, due to some MPI-specific tests in our examples, we prefer our [testflo](https://github.com/OpenMDAO/testflo) package.
The testflo utility can be installed using

```
python -m pip install testflo
```

Testflo can be invoked from the top-level Dymos directory with:

```
testflo .
```

With pyoptsparse correctly installed and things working correctly, the tests should conclude after several minutes with a message like the following:
The lack of MPI capability or pyoptsparse will cause additional tests to be skipped.

```
The following tests were skipped:
test_command_line.py:TestCommandLine.test_ex_brachistochrone_reset_grid


OK


Passed:  450
Failed:  0
Skipped: 1

Ran 451 tests using 2 processes
```
