import pandas as pd
from scipy.optimize import minimize


class ModelCalibration:
    DEFAULT_PARAMETER_LOWER_BOUND = 1e-4

    def __init__(self, options):
        self.options = options

    def calibrate_model(self, model, method=None, options=None, bounds='default'):
        if bounds == 'default':
            bounds = ((self.DEFAULT_PARAMETER_LOWER_BOUND, None),) * len(model.parameters)
        loss = self._get_loss_function(model)
        res = minimize(loss, model.parameters, bounds=bounds, method=method, options=options)
        model.parameters = res.x
        return res, model

    def _get_loss_function(self, model):
        def _loss_function(parameters):
            model.parameters = parameters
            predicted_prices = pd.Series([model.price_option_black(o) for o in self.options])
            real_prices = pd.Series([o.price for o in self.options])
            return sum((predicted_prices - real_prices) ** 2)

        return _loss_function
