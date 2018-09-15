*****************
Model calibration
*****************

In the context of this package, calibration is a procedure which takes an option pricing model
and a set of listed vanilla options and tunes the parameters of the model so that the option price
predicted by the model is as close as possible to the actual prices of listed options.

Creating inputs
===============

We'll suppose that our training set has only three options.

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_princing.option import VanillaOption
    from vanilla_option_princing.models import OrnsteinUhlenbeck
    from vanilla_option_pricing.calibration import ModelCalibration

    data_set = [
        VanillaOption('TTF', 'c', date(2018, 1, 1), 2, 101, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'p', date(2018, 1, 1), 2, 98, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'c', date(2018, 1, 1), 5, 101, 100, date(2018, 5, 31))
    ]

We want to calibrate both a Black-Scholes and an Ornstein-Uhlenbeck model.

.. code:: python

    models = [
        BlackScholes(0.2),
        OrnsteinUhlenbeck(p_0=0, l=100, s=2)
    ]


Calibrating models
==================

We can now instantiate the calibration object, run the optimization algorithm and inspect the results.

.. code:: python

    calibration = ModelCalibration(data_set)

    for model in models:
        option_pricing_model = model.as_option_pricing_model()
        result, trained_model = calibration.calibrate_model(option_pricing_model)
        print('Optimization results:')
        print(result)
        print(f'Calibrated parameters: {trained_model.parameters}\n\n')