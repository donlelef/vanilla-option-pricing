from datetime import datetime, timedelta

import numpy as np
import pytest
from pytest import fixture

from vanilla_option_pricing.option import VanillaOption, option_list_to_pandas_dataframe, \
    pandas_dataframe_to_option_list


@fixture
def option():
    return VanillaOption(
        spot=100,
        strike=100,
        dividend=1,
        date=datetime.today(),
        maturity=datetime.today() + timedelta(days=30),
        option_type='c',
        price=1,
        instrument='miao'
    )


@fixture
def option_list():
    return [
        VanillaOption(spot=100, strike=100, dividend=1, date=datetime.today(), maturity=datetime.today(),
                      option_type='c', price=1, instrument='miao'),
        VanillaOption(spot=200, strike=200, dividend=2, date=datetime.today(), maturity=datetime.today(),
                      option_type='p', price=2, instrument='bau')
    ]


def test_years_to_maturity(option: VanillaOption):
    assert option.years_to_maturity == pytest.approx(30 / 365.25, abs=0.01)


def test_implied_volatility(option: VanillaOption):
    assert option.implied_volatility_of_undiscounted_price == pytest.approx(0.0875, abs=0.0001)


def test_to_dict(option: VanillaOption):
    assert option.to_dict() == {
        'spot': option.spot,
        'strike': option.strike,
        'dividend': option.dividend,
        'date': option.date,
        'maturity': option.maturity,
        'option_type': option.option_type,
        'price': option.price,
        'instrument': option.instrument
    }


def test_error_on_invalid_option_type(option: VanillaOption):
    option.option_type = 'x'
    with pytest.raises(ValueError) as err:
        _ = option.implied_volatility_of_undiscounted_price
        assert 'option_type shall be' in err.value


def test_option_list_to_pandas_dataframe(option_list):
    data_frame = option_list_to_pandas_dataframe(option_list)
    assert (2, 8) == data_frame.shape
    np.testing.assert_array_equal(np.array([100, 200]), data_frame.spot)


def test_pandas_dataframe_to_option_list(option_list):
    data_frame = option_list_to_pandas_dataframe(option_list)
    options = pandas_dataframe_to_option_list(data_frame)
    assert len(options) == 2
    for i in range(len(options)):
        assert type(options[i]) == VanillaOption
    assert options[0].instrument == 'miao'
    assert options[1].instrument == 'bau'
