from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np

from vanilla_option_pricing.models.models import LogMeanRevertingToGeneralisedWienerProcess, OrnsteinUhlenbeck, \
    BlackScholes, NumericalLogMeanRevertingToGeneralisedWienerProcess, NumericalModel
from vanilla_option_pricing.option import VanillaOption


class TestOrnsteinUhlenbeck(TestCase):

    def setUp(self):
        p_0 = 1
        self.l = l = 1
        self.s = s = 1
        self.ou_model = OrnsteinUhlenbeck(p_0, l, s).as_option_pricing_model()
        self.test_option = VanillaOption(
            spot=1.5,
            strike=1,
            dividend=0.02,
            date=datetime.today(),
            maturity=datetime.today() + timedelta(days=2 * 365.2425),
            option_type='c',
            price=0,
            instrument='miao'
        )

    def test_variance(self):
        expected_res = 0.5676676416183064
        self.assertAlmostEqual(expected_res, self.ou_model.variance(1), places=5)

    def test_std_deviation(self):
        expected_res = np.sqrt(0.5676676416183064)
        self.assertAlmostEqual(expected_res, self.ou_model.standard_deviation(1), places=5)

    def test_black_scholes_merton_follows_equation(self):
        expected_res = 0.6581200069766773
        res = self.ou_model.price_option_black_scholes_merton(self.test_option, risk_free_rate=0.05)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_black_follows_equation(self):
        expected_res = 0.650051451659
        res = self.ou_model.price_option_black(self.test_option)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_properties(self):
        self.assertEqual([self.l, self.s], self.ou_model.model.parameters)


class TestLogMeanRevertingToGeneralisedWienerProcess(TestCase):

    def setUp(self):
        p_0 = np.matrix(np.identity(2))
        self.l = 1
        self.s_x = 1
        self.s_y = 0.05
        self.model = LogMeanRevertingToGeneralisedWienerProcess(p_0, self.l, self.s_x, self.s_y)

    def test_variance(self):
        expected_res = 0.967664270613846
        res = self.model.variance(1)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_properties(self):
        self.assertEqual([self.l, self.s_x, self.s_y], self.model.parameters)


class TestNumericalLogMeanRevertingToGeneralisedWienerProcess(TestCase):

    def setUp(self):
        p_0 = np.matrix(np.identity(2))
        self.l = 1
        self.s_x = 1
        self.s_y = 0.05
        self.model = NumericalLogMeanRevertingToGeneralisedWienerProcess(p_0, self.l, self.s_x, self.s_y)

    def test_variance(self):
        expected_res = 0.967664270613846
        res = self.model.variance(1)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_properties(self):
        self.assertEqual([self.l, self.s_x, self.s_y], self.model.parameters)


class TestBlackScholes(TestCase):

    def setUp(self):
        self.s = 2
        self.model = BlackScholes(self.s)

    def test_variance(self):
        expected_res = 2 ** 2 * 1
        res = self.model.variance(1)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_properties(self):
        self.assertEqual([self.s], self.model.parameters)


class TestNumericalModel(TestCase):

    def setUp(self):
        self.A = np.matrix(1)
        self.B = np.matrix(0)
        self.p_0 = np.matrix(1)
        self.model = NumericalModel(self.A, self.B, self.p_0)

    def test_variance(self):
        expected_res = np.exp(2)
        res = self.model.variance(1)
        self.assertAlmostEqual(expected_res, res, places=5)

    def test_properties(self):
        self.assertEqual([self.A, self.B], self.model.parameters)
