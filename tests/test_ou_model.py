from datetime import datetime, timedelta

import numpy as np
import pytest
from pytest import fixture

from vanilla_option_pricing.models import OrnsteinUhlenbeck
from vanilla_option_pricing.option import VanillaOption


@fixture
def option():
    return VanillaOption(
        spot=1.5,
        strike=1,
        dividend=0.02,
        date=datetime.today(),
        maturity=datetime.today() + timedelta(days=2 * 365.2425),
        option_type='c',
        price=0,
        instrument='miao'
    )


@fixture
def model():
    return OrnsteinUhlenbeck(1, 1, 1).as_option_pricing_model()


def test_variance(model):
    assert model.variance(1) == pytest.approx(0.5676676416183064, abs=1e-4)


def test_std_deviation(model):
    assert model.standard_deviation(1) == pytest.approx(np.sqrt(0.5676676416183064), abs=1e-4)


def test_black_scholes_merton(model, option):
    assert model.price_option_black_scholes_merton(option, 0.05) == pytest.approx(0.6581200069766773, abs=1e-4)


def test_black(model, option):
    assert model.price_option_black(option) == pytest.approx(0.650051451659, abs=1e-4)


def test_properties(model):
    assert model.parameters == (1, 1)
