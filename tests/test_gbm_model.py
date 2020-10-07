from pytest import fixture

from vanilla_option_pricing.models import BlackScholes


@fixture
def model():
    return BlackScholes(2).as_option_pricing_model()


def test_variance(model):
    assert model.variance(1) == 4


def test_properties(model):
    assert model.parameters == (2,)
