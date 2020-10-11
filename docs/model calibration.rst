*****************
Model calibration
*****************

The market calibration calibration is a procedure which takes an option pricing model
and a set of listed vanilla options and tunes the parameters of the model so that the option price
given by the model is as close as possible to the actual prices of listed options.

More rigorous details and a mathematical formulation can be found in the paper
`Fast calibration of two-factor models for energy option pricing <https://arxiv.org/abs/1809.03941>`_.

Creating inputs
===============

We'll suppose that the available dataset to tune our model contains has only three options. In a realistic
scenario, tens to hundreds of options would be needed.

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_princing.option import VanillaOption
    from vanilla_option_princing.models import GeometricBrownianMotion, OrnsteinUhlenbeck
    from vanilla_option_pricing.calibration import ModelCalibration

    data_set = [
        VanillaOption('TTF', 'c', date(2018, 1, 1), 2, 101, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'p', date(2018, 1, 1), 2, 98, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'c', date(2018, 1, 1), 5, 101, 100, date(2018, 5, 31))
    ]

We want to calibrate both a Geometric Brownian motion and an Ornstein-Uhlenbeck model.

.. code:: python

    models = [
        GeometricBrownianMotion(0.2),
        OrnsteinUhlenbeck(p_0=0, l=100, s=2)
    ]


Calibrating models
==================

We can now instantiate the calibration object, run the optimization algorithm and inspect the results.

.. code:: python

    calibration = ModelCalibration(data_set)

    for model in models:
        result, trained_model = calibration.calibrate_model(model)
        print('Optimization results:')
        print(result)
        print(f'Calibrated parameters: {trained_model.parameters}\n\n')