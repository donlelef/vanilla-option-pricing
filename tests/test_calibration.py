from pathlib import Path

import pandas as pd
import pytest
from pytest import fixture

from vanilla_option_pricing.calibration import ModelCalibration
from vanilla_option_pricing.models import GeometricBrownianMotion
from vanilla_option_pricing.option import pandas_dataframe_to_option_list


@fixture
def option_list():
    path = Path(__file__).resolve().parent / 'test_data' / 'data.csv'
    data_set = pd.read_csv(path, parse_dates=['date', 'maturity'], dayfirst=True)
    return pandas_dataframe_to_option_list(data_set)


def test_calibrate_black_model(option_list):
    calibrator = ModelCalibration(option_list)
    model = GeometricBrownianMotion(0.31408317454633633)
    res, model = calibrator.calibrate_model(model)
    assert res.x[0] == pytest.approx(0.26914529578104857, abs=10e-4)


def test_calibrate_makes_deepcopy(option_list):
    calibrator = ModelCalibration(option_list)
    model = GeometricBrownianMotion(0.31408317454633633)
    res, new_model = calibrator.calibrate_model(model)
    assert new_model.s == pytest.approx(0.26914529578104857, abs=10e-4)
    assert model.s == pytest.approx(0.31408317454633633, abs=10e-4)
