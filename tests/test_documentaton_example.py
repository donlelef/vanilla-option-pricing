from datetime import datetime, timedelta, date

import numpy as np
import pytest
from pytest import fixture

from vanilla_option_pricing.calibration import ModelCalibration
from vanilla_option_pricing.models import GeometricBrownianMotion, OrnsteinUhlenbeck, \
    LogMeanRevertingToGeneralisedWienerProcess
from vanilla_option_pricing.option import VanillaOption


@fixture
def option():
    return VanillaOption(
        spot=100,
        strike=101,
        dividend=0,
        date=datetime.today(),
        maturity=datetime.today() + timedelta(days=30),
        option_type='c',
        price=1,
        instrument='TTF'
    )


def test_model_execution(option):
    volatility = option.implied_volatility_of_undiscounted_price
    assert volatility == pytest.approx(0.12578680488787206, abs=10e-4)

    t = 0.5
    bs_model = GeometricBrownianMotion(volatility)
    assert bs_model.volatility(t) == pytest.approx(0.12578680488787203, abs=10e-4)

    ou_model = OrnsteinUhlenbeck(0, 100, 0.2)
    assert ou_model.volatility(t) == 0.02

    lmrgw_model = LogMeanRevertingToGeneralisedWienerProcess(np.array([[0, 0], [0, 0]]), 100, 0.1, 0.3)
    assert lmrgw_model.volatility(t) == pytest.approx(0.2956349099818896, abs=1e-4)


def test_calibration():
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
