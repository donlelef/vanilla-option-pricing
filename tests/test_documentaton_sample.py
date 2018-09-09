from datetime import datetime, timedelta, date
from unittest import TestCase

from vanilla_option_pricing.calibration import ModelCalibration
from vanilla_option_pricing.models import BlackScholes
from vanilla_option_pricing.option import VanillaOption


class TestVanillaOption(TestCase):

    def test_option_creation(self):
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
        model = BlackScholes(volatility).as_option_pricing_model()
        model_price = model.price_option_black(option)
        print(f'Actual price: {option.price}, model price: {model_price}')
        self.assertAlmostEqual(option.price, model_price, places=4)

    def test_calibration(self):
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
