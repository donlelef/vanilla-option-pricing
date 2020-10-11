from typing import Dict, Sequence

import pytest


def check_exception_on_wrong_parameters(
        model_class: type,
        illegal_constructor_params: Dict[str, float],
        legal_constructor_params: Dict[str, float],
        setter_params: Sequence[float]):
    with pytest.raises(ValueError) as err:
        model = model_class(**illegal_constructor_params)
    assert "must be non-negative" in err.value.args[0]
    with pytest.raises(ValueError) as err:
        model = model_class(**legal_constructor_params)
        model.parameters = setter_params
    assert "must be non-negative" in err.value.args[0]
