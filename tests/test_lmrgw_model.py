import numpy as np
import pytest
from pytest import fixture

from vanilla_option_pricing.models import LogMeanRevertingToGeneralisedWienerProcess, \
    NumericalLogMeanRevertingToGeneralisedWienerProcess


@fixture
def model():
    p_0 = np.eye(2)
    return LogMeanRevertingToGeneralisedWienerProcess(p_0, 1, 1, 0.05)


@fixture
def numerical_model():
    p_0 = np.eye(2)
    return NumericalLogMeanRevertingToGeneralisedWienerProcess(p_0, 1, 1, 0.05)


def test_variance(model):
    assert model.variance(1) == pytest.approx(0.967664270613846, abs=10e-4)


def test_properties(numerical_model):
    assert numerical_model.parameters == (1, 1, 0.05)


def test_numerical_variance(numerical_model):
    assert numerical_model.variance(1) == pytest.approx(0.967664270613846, abs=10e-4)


def test_numerical_properties(model):
    assert model.parameters == (1, 1, 0.05)
