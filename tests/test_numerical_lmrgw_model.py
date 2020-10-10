import numpy as np
import pytest
from pytest import fixture

from vanilla_option_pricing.models import NumericalModel


@fixture
def model():
    return NumericalModel(np.array([1]), np.array([0]), np.array([1]))


def test_variance(model):
    assert model.variance(1) == pytest.approx(np.exp(2), abs=10e-4)


def test_properties(model):
    assert model.A == 1
    assert model.B == 0
