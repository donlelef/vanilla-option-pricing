from pytest import fixture

from vanilla_option_pricing.models import GeometricBrownianMotion


@fixture
def model():
    return GeometricBrownianMotion(2)


def test_variance(model):
    assert model.variance(1) == 4


def test_properties(model):
    assert model.parameters == (2,)
