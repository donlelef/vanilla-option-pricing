***************
Getting started
***************

This tutorial shows basic usage of this package.

Creating options
================

Let's create a sample call option

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_princing.option import VanillaOption
    from vanilla_option_princing.models import GeometricBrownianMotion
    from vanilla_option_pricing.calibration import ModelCalibration

    option = VanillaOption(
        spot=100,
        strike=101,
        dividend=0,
        date=datetime.today(),
        maturity=datetime.today() + timedelta(days=30),
        option_type='c',
        price=1,
        instrument='TTF'
    )

Implied volatility and option pricing
=====================================

We can compute the implied volatility and create a Geometric Brownian Motion
model with it. Of course, if now we ask price the option using the Black framework,
we'll get back the initial price.

.. code:: python

    volatility = option.implied_volatility_of_undiscounted_price
    model = GeometricBrownianMotion(volatility)
    model_price = model.price_option_black(option)
    print(f'Actual price: {option.price}, model price: {model_price}')

Calibrating models
==================

We can also try and calibrate the parameters of a model against
listed options.

.. code:: python

    data_set = [
        VanillaOption('TTF', 'c', date(2018, 1, 1), 2, 101, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'p', date(2018, 1, 1), 2, 98, 100, date(2018, 2, 1)),
        VanillaOption('TTF', 'c', date(2018, 1, 1), 5, 101, 100, date(2018, 5, 31))
    ]

    for o in data_set:
        print(f'Implied volatility: {o.implied_volatility_of_undiscounted_price}')

    model = GeometricBrownianMotion(0.2)
    calibration = ModelCalibration(data_set)

    result, tuned_model = calibration.calibrate_model(model)
    print(result)
    print(f'Calibrated implied volatility: {tuned_model.parameters[0]}')

As we can see, the calibration process takes the implied volatility of the model close
to the average of the options it has been trained on.




