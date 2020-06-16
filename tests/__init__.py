import theano

import pymc3


# Prevent slow C compilation
# theano.config.compute_test_value = "ignore"
theano.config.on_opt_error = "raise"
theano.config.mode = "FAST_COMPILE"
theano.config.cxx = ""
