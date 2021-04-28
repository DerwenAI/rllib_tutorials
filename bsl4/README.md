# Unresolved Bugs


##  NameError `_C`

Reproducing the the `NameError: name '_C' is not defined"` exception
with Ray 1.2.0 on macOS:

   * `requirements.txt` -- dependencies installed
   * `libs.txt` -- pip freeze output of dependency versions
   * `cpv1.ipynb` -- notebook code
   * `cpv1.py` -- same code run from command line
   * `out.txt` -- output from successful run via command line

```
python3 cpv1.py > out.txt 2>&1
```

Given the same code and same environment:

  * executed as a Python script from the command line run successfully
  * however, running within a JupyterLab notebook throws `NameError`

There are various GitHub issues for different projects which describe
this error (not with Ray) and discuss bumping NumPy to 1.19.3 or
later, although here NumPy is version 1.19.5


---

## Caveats

There is a known issue with JupyterLab + Conda, where the behavior
differs from running the same notebook with Jupyter.

A JupyterLab kernel may be running a different Python executable than
scripts launched from the same command line.

The `conda` installer is aggressive about changing the `PATH`
environment variable -- check that and reset `PATH` if needed.

See https://github.com/explosion/spaCy/discussions/7435 for details.
