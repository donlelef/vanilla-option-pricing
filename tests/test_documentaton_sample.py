from datetime import datetime, timedelta, date
from unittest import TestCase

import numpy as np

from vanilla_option_pricing.calibration import ModelCalibration
from vanilla_option_pricing.models import BlackScholes, OrnsteinUhlenbeck, \
    LogMeanRevertingToGeneralisedWienerProcess
from vanilla_option_pricing.option import VanillaOption


class TestVanillaOption(TestCase):

    def setUp(self):
        self.option = VanillaOption(
            spot=100,
            strike=101,
            dividend=0,
            date=datetime.today(),
            maturity=datetime.today() + timedelta(days=30),
            option_type='c',
            price=1,
            instrument='TTF'
        )

    def test_option_creation(self):
        option = self.option
        volatility = option.implied_volatility_of_undiscounted_price
        print(f'Option volatility is {volatility}')

    def test_models_execution(self):
        option = self.option
        volatility = option.implied_volatility_of_undiscounted_price
        print(f'Option volatility is {volatility}')
        bs_model = BlackScholes(volatility).as_option_pricing_model()
        t = 0.5
        print(f'At time t={t} years, volatility is {bs_model.volatility(t)}, '
              f'variance is {bs_model.variance(t)}, '
              f'standard deviation is {bs_model.standard_deviation(t)}')

        ou_model = OrnsteinUhlenbeck(0, 100, 0.2).as_option_pricing_model()
        print(f'At time t={t} years, volatility is {ou_model.volatility(t)}, '
              f'variance is {ou_model.variance(t)}, '
              f'standard deviation is {ou_model.standard_deviation(t)}')

        lmrgw_model = LogMeanRevertingToGeneralisedWienerProcess(
            np.matrix([[0, 0], [0, 0]]), 100, 0.1, 0.3
        ).as_option_pricing_model()
        print(f'At time t={t} years, volatility is {lmrgw_model.volatility(t)}, '
              f'variance is {lmrgw_model.variance(t)}, '
              f'standard deviation is {lmrgw_model.standard_deviation(t)}')

    def test_calibration(self):
        data_set = [
            VanillaOption('TTF', 'c', date(2018, 1, 1), 2, 101, 100, date(2018, 2, 1)),
            VanillaOption('TTF', 'p', date(2018, 1, 1), 2, 98, 100, date(2018, 2, 1)),
            VanillaOption('TTF', 'c', date(2018, 1, 1), 5, 101, 100, date(2018, 5, 31))
        ]

        models = [
            BlackScholes(0.2),
            OrnsteinUhlenbeck(p_0=0, l=100, s=2)
        ]
        calibration = ModelCalibration(data_set)

        print(f'Implied volatilities: {[o.implied_volatility_of_undiscounted_price for o in data_set]}\n')

        for model in models:
            option_pricing_model = model.as_option_pricing_model()
            result, trained_model = calibration.calibrate_model(option_pricing_model)
            print('Optimization results:')
            print(result)
            print(f'Calibrated parameters: {trained_model.parameters}\n\n')
