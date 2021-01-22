![Build Status](https://github.com/AmpersandTV/pymc3-hmm/workflows/PyMC3-HMM/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AmpersandTV/pymc3-hmm/main?filepath=examples)

# PyMC3 HMM

Hidden Markov models in [PyMC3](https://github.com/pymc-devs/pymc3).

### Features
- Fully implemented PyMC3 `Distribution` classes for HMM state sequences (`DiscreteMarkovChain`) and mixtures that are driven by them (`SwitchingProcess`)
- A forward-filtering backward-sampling (FFBS) implementation (`FFBSStep`) that works with NUTS&mdash;or any other PyMC3 sampler
- A conjugate Dirichlet transition matrix sampler (`TransMatConjugateStep`)
- Support for time-varying transition matrices in the FFBS sampler and all the relevant `Distribution` classes

To use these distributions and step methods in your PyMC3 models, simply import them from the `pymc3_hmm` package.

See the [examples directory](https://nbviewer.jupyter.org/github/AmpersandTV/pymc3-hmm/tree/main/examples/) for demonstrations of the aforementioned features.  You can also use [Binder](https://mybinder.org/v2/gh/AmpersandTV/pymc3-hmm/main?filepath=examples) to run the examples yourself.

## Installation

Currently, the package can be installed via `pip` directly from GitHub
```shell
$ pip install git+https://github.com/AmpersandTV/pymc3-hmm
```

## Development

First, pull in the source from GitHub:

```python
$ git clone git@github.com:AmpersandTV/pymc3-hmm.git
```

Next, you can run `make conda` or `make venv` to set up a virtual environment.

Once your virtual environment is set up, install the project, its dependencies, and the `pre-commit` hooks:

```bash
$ pip install -r requirements.txt
$ pre-commit install --install-hooks
```



After making changes, be sure to run `make black` in order to automatically format the code and then `make check` to run the linters and tests.

## License

[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
