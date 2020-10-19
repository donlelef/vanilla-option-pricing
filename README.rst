.. role:: raw-html-m2r(raw)
   :format: html


Vanilla Option Pricing
======================


.. image:: https://github.com/donlelef/vanilla-option-pricing/workflows/Python%20package/badge.svg
   :target: https://github.com/donlelef/vanilla-option-pricing/actions
   :alt: Actions Status


.. image:: https://codecov.io/gh/donlelef/vanilla-option-pricing/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/donlelef/vanilla-option-pricing
   :alt: codecov


.. image:: https://readthedocs.org/projects/vanilla-option-pricing/badge/?version=latest
   :target: https://vanilla-option-pricing.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://pepy.tech/badge/vanilla-option-pricing
   :target: https://pepy.tech/project/vanilla-option-pricing
   :alt: Downloads


.. image:: https://zenodo.org/badge/147844047.svg
   :target: https://zenodo.org/badge/latestdoi/147844047
   :alt: DOI

A Python package implementing stochastic models to price financial options.\ :raw-html-m2r:`<br>`
The theoretical background and a comprehensive explanation of models and their parameters
can be found is the paper *\ `Fast calibration of two-factor models for energy option pricing <https://arxiv.org/abs/1809.03941>`_\ *
by Emanuele Fabbiani, Andrea Marziali and Giuseppe De Nicolao, freely available on arXiv.  

Installing
^^^^^^^^^^

The preferred way to install the package is using pip,
but you can also download the code and install from source

To install the package using pip:

.. code-block:: bash

   pip install vanilla_option_pricing

Quickstart
^^^^^^^^^^

Let's create a call option.

.. code-block:: python

   from datetime import datetime, timedelta
   from vanilla_option_pricing.option import VanillaOption

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

We can compute the implied volatility and create a Geometric Brownian Motion 
model with it. Of course, if now we ask price the option using the Black framework, 
we'll get back the initial price.

.. code-block:: python

   from vanilla_option_pricing.models import GeometricBrownianMotion

   volatility = option.implied_volatility_of_undiscounted_price
   gbm_model = GeometricBrownianMotion(volatility)
   gbm_price = gbm_model.price_option_black(option)
   print(f'Actual price: {option.price}, model price: {gbm_price}')

But, if we adopt a different model, say a Log-spot price mean reverting to 
generalised Wiener process model (MLR-GW), we will get a different price.

.. code-block:: python

   import numpy as np
   from vanilla_option_pricing.models import LogMeanRevertingToGeneralisedWienerProcess

   p_0 = np.eye(2)
   model = LogMeanRevertingToGeneralisedWienerProcess(p_0, 1, 1, 1)
   lmrgw_price = model.price_option_black(option)
   print(f'Actual price: {option.price}, model price: {lmrgw_price}')

In the previous snippet, the parameters of the LMR-GW model were chosen
at random. We can also calibrate the parameters of a model against 
listed options.

.. code-block:: python

   from datetime import date
   from vanilla_option_pricing.option import VanillaOption
   from vanilla_option_pricing.models import OrnsteinUhlenbeck, GeometricBrownianMotion
   from vanilla_option_pricing.calibration import ModelCalibration

   data_set = [
       VanillaOption('TTF', 'c', date(2018, 1, 1), 2, 101, 100, date(2018, 2, 1)),
       VanillaOption('TTF', 'p', date(2018, 1, 1), 2, 98, 100, date(2018, 2, 1)),
       VanillaOption('TTF', 'c', date(2018, 1, 1), 5, 101, 100, date(2018, 5, 31))
   ]

   models = [
       GeometricBrownianMotion(0.2),
       OrnsteinUhlenbeck(p_0=0, l=100, s=2)
   ]
   calibration = ModelCalibration(data_set)

   print(f'Implied volatilities: {[o.implied_volatility_of_undiscounted_price for o in data_set]}\n')

   for model in models:
       result, trained_model = calibration.calibrate_model(model)
       print('Optimization results:')
       print(result)
       print(f'Calibrated parameters: {trained_model.parameters}\n\n')
