import pandas as pd
from scipy.optimize import minimize


class ModelCalibration:
    DEFAULT_PARAMETER_LOWER_BOUND = 1e-4

    def __init__(self, options, pricing_model):
        self.options = options
        self.model = pricing_model

    def calibrate_model(self, method=None, options=None, bounds='default'):
        if bounds == 'default':
            bounds = ((self.DEFAULT_PARAMETER_LOWER_BOUND, None),) * len(self.model.parameters)
        res = minimize(self._loss_function,
                       self.model.parameters,
                       bounds=bounds, method=method, options=options)
        self.model.parameters = res.x
        return res, self.model

    def _loss_function(self, parameters):
        self.model.parameters = parameters
        predicted_prices = pd.Series([self.model.price_option_black(o) for o in self.options])
        real_prices = pd.Series([o.price for o in self.options])
        return sum((predicted_prices - real_prices) ** 2)
