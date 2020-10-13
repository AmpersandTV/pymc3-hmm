![Build Status](https://github.com/AmpersandTV/pymc3-hmm/workflows/PyMC3-HMM/badge.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AmpersandTV/pymc3-hmm/main?filepath=examples)

# PyMC3 HMM

Hidden Markov models in [PyMC3](https://github.com/pymc-devs/pymc3).

### Features
- Fully implemented PyMC3 `Distribution` classes for HMM state sequences and mixtures that depend on them
- A forward-filtering backward-sampling (FFBS) implementation that works with NUTS&mdash;or any other PyMC3 sampler
- A conjugate Dirichlet transition matrix sampler
- Support for time-varying transition matrices in both the `Distribution` classes and FFBS sampler


See the [examples directory](https://nbviewer.jupyter.org/github/AmpersandTV/pymc3-hmm/tree/main/examples/) for demonstrations of the aforementioned features.  You can also use [Binder](https://mybinder.org/v2/gh/AmpersandTV/pymc3-hmm/main?filepath=examples) to run the examples yourself.

## Installation

The package name is `pymc3_hmm` and it can be installed with `pip` directly from GitHub
```shell
$ pip install git+https://github.com/AmpersandTV/pymc3-hmm
```

## Development

First, pull in the source from GitHub:

```python
$ git clone git@github.com:AmpersandTV/pymc3-hmm.git
```

Afterward, you can run `make conda` or `make venv` to set up a virtual environment.  After making changes, be sure to run `make black` in order to automatically format the code and then `make check` to run the linters and tests.

## License

[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
