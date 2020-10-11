******
Models
******

In the context of this package, a model is a stochastic process.

The package APIs offer a simple way of extracting the variance, the standard deviation and the volatility
derived from a model at a given time instant.

There are three models currently implemented by this package: a detailed description and further references can
be found in the paper
`Fast calibration of two-factor models for energy option pricing <https://arxiv.org/abs/1809.03941>`_.

Geometric Brownian Motion
=========================

The celebrated Geometric Brownian Motion process, adopted in the Black and Black-Scholes-Merton frameworks for
option pricing.

.. code:: python

    from datetime import datetime, timedelta
    from vanilla_option_pricing.option import VanillaOption
    from vanilla_option_pricing.models import GeometricBrownianMotion

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

    volatility = option.implied_volatility_of_undiscounted_price
    print(f'Option volatility is {volatility}')
    gbm_model = GeometricBrownianMotion(volatility)
    t = 0.5
    print(f'At time t={t} years, volatility is {gbm_model.volatility(t)}, '
          f'variance is {gbm_model.variance(t)}, '
          f'standard deviation is {gbm_model.standard_deviation(t)}')


Ornstein-Uhlenbeck
==================

The Ornstein-Uhlenbeck process, the simplest mean-reverting model, quite popular for energy commodities.

.. code:: python

    from vanilla_option_pricing.models import OrnsteinUhlenbeck

    ou_model = OrnsteinUhlenbeck(
        p_0 = 1,
        l = 1,
        s = volatility
    )
    print(f'At time t={t} years, volatility is {ou_model.volatility(t)}, '
          f'variance is {ou_model.variance(t)}, '
          f'standard deviation is {ou_model.standard_deviation(t)}')


Log Mean-Reverting To Generalised Wiener Process
================================================

One of the most common two-factor, mean-reverting models.

.. code:: python

    import numpy as np
    from vanilla_option_pricing.models import LogMeanRevertingToGeneralisedWienerProcess

    lmrgw_model = LogMeanRevertingToGeneralisedWienerProcess(
        p_0 = np.eye(2),
        l = 100,
        s_x = 0.1,
        s_y = 0.3
    )
    print(f'At time t={t} years, volatility is {lmrgw_model.volatility(t)}, '
          f'variance is {lmrgw_model.variance(t)}, '
          f'standard deviation is {lmrgw_model.standard_deviation(t)}')

