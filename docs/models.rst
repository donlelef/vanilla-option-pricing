******
Models
******

In the context of this package, a model is a stochastic process.
The package APIs offer a simple way of extracting the variance, the standard deviation and the volatility
derived from a model at a given time instant.
A decorator takes care of giving each model option pricing capabilities.
There are three models currently implemented by this package.

Black-Scholes
=============

The famous Black-Sholes model, based on the assumption that the log-spot price of the underlying behaves as a
Geometric Brownian motion.

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_pricing.models import BlackScholes, OrnsteinUhlenbeck, \
        LogMeanRevertingToGeneralisedWienerProcess
    from vanilla_option_pricing.option import VanillaOption

    option = VanillaOption(spot=100, strike=101, dividend=0,
        date=datetime.today(), maturity=datetime.today() + timedelta(days=30),
        option_type='c', price=1, instrument='TTF'
    )

    volatility = option.implied_volatility_of_undiscounted_price
    print(f'Option volatility is {volatility}')
    bs_model = BlackScholes(volatility).as_option_pricing_model()
    t = 0.5
    print(f'At time t={t} years, volatility is {bs_model.volatility(t)}, '
          f'variance is {bs_model.variance(t)}, '
          f'standard deviation is {bs_model.standard_deviation(t)}')


Ornstein-Uhlenbeck
==================

The most simple single-factor, mean-reverting model.

.. code:: python

    bs_model = BlackScholes(volatility).as_option_pricing_model()
    print(f'At time t={t} years, volatility is {bs_model.volatility(t)}, '
          f'variance is {bs_model.variance(t)}, '
          f'standard deviation is {bs_model.standard_deviation(t)}')


Log Mean-Reverting To Generalised Wiener Process
================================================

One of the most common two-factor, mean-reverting models.

.. code:: python

    lmrgw_model = LogMeanRevertingToGeneralisedWienerProcess(
            np.matrix([[0, 0], [0, 0]]), 100, 0.1, 0.3
        ).as_option_pricing_model()
    print(f'At time t={t} years, volatility is {lmrgw_model.volatility(t)}, '
          f'variance is {lmrgw_model.variance(t)}, '
          f'standard deviation is {lmrgw_model.standard_deviation(t)}')

