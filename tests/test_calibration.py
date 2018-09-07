import os
from unittest import TestCase

import pandas as pd

from vanilla_option_pricing.calibration import ModelCalibration
from vanilla_option_pricing.models.models import BlackScholes
from vanilla_option_pricing.option import VanillaOption


class TestModelCalibration(TestCase):
    def setUp(self):
        file_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'fixtures', 'data.csv'))
        dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
        data_set = pd.read_csv(file_path, parse_dates=['date', 'maturity'], date_parser=dateparse)
        self.options = [VanillaOption(**r) for r in data_set.to_dict(orient='record')]

    def test_calibrate_black_model(self):
        calibrator = ModelCalibration(self.options, BlackScholes(0.31408317454633633).as_option_pricing_model())
        res, model = calibrator.calibrate_model()
        self.assertAlmostEqual(res.x[0], 0.26914529578104857, places=5)
        self.assertAlmostEqual(model.model.s, 0.26914529578104857, places=5)
