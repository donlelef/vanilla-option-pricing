from datetime import datetime, timedelta
from unittest import TestCase

from vanilla_option_pricing.option import VanillaOption, option_list_to_pandas_dataframe, \
    pandas_dataframe_to_option_list


class TestVanillaOption(TestCase):

    def setUp(self):
        self.option = VanillaOption(
            spot=100,
            strike=100,
            dividend=1,
            date=datetime.today(),
            maturity=datetime.today() + timedelta(days=30),
            option_type='c',
            price=1,
            instrument='miao'
        )

    def test_years_to_maturity(self):
        self.assertAlmostEqual(30 / 365.25, self.option.years_to_maturity, places=2)

    def test_implied_volatility(self):
        self.assertAlmostEqual(self.option.implied_volatility_of_undiscounted_price, 0.0875, places=4)


class TestUtilsFunctions(TestCase):

    def setUp(self):
        self.options_list = [
            VanillaOption(spot=100, strike=100, dividend=1, date=datetime.today(), maturity=datetime.today(),
                          option_type='c', price=1, instrument='miao')
        ]

    def test_option_list_to_pandas_dataframe(self):
        data_frame = option_list_to_pandas_dataframe(self.options_list)
        self.assertEqual((1, 8), data_frame.shape)
        self.assertEqual('miao', data_frame.instrument[0])

    def test_pandas_dataframe_to_option_list(self):
        data_frame = option_list_to_pandas_dataframe(self.options_list)
        options = pandas_dataframe_to_option_list(data_frame)
        self.assertEqual(1, len(options))
        self.assertEqual(VanillaOption, type(options[0]))
        self.assertEqual('miao', options[0].instrument)
