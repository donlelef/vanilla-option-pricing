***************
Getting started
***************

This tutorial is meant to show basic usage of this package.

Creating options
================

Let's create a sample call option

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_princing.option import VanillaOption
    from vanilla_option_princing.models import BlackScholes
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

We can compute the implied volatility and create a Black-Sholes model
with it. Of course, if now we ask the model to price the option, we'll
get the real option price.

.. code:: python

    volatility = option.implied_volatility_of_undiscounted_price
    model = BlackScholes(volatility).as_option_pricing_model()
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

    model = BlackScholes(0.2).as_option_pricing_model()
    calibration = ModelCalibration(data_set)

    result, trained_model = calibration.calibrate_model(model)
    print(result)
    print(f'Calibrated implied volatility: {trained_model.parameters[0]}')

As we can see, the calibration process takes the implied volatility of the model close
to the average of the options it has been trained on.




