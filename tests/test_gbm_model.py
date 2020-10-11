from pytest import fixture

from tests.utils import check_exception_on_wrong_parameters
from vanilla_option_pricing.models import GeometricBrownianMotion


@fixture
def model():
    return GeometricBrownianMotion(2)


def test_variance(model):
    assert model.variance(1) == 4


def test_exception_on_illegal_parameters():
    check_exception_on_wrong_parameters(GeometricBrownianMotion, {'s': -1}, {'s': 1}, (-1,))


def test_properties(model):
    assert model.parameters == (2,)
