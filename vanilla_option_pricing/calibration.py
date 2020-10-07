from typing import Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from vanilla_option_pricing.option_pricing import OptionPricingModel


class ModelCalibration:
    """
    Calibrate option pricing models according to prices of listed options

    :param options: a collection of :class:`~option.VanillaOption`
    """

    DEFAULT_PARAMETER_LOWER_BOUND = 1e-4

    def __init__(self, options):
        self.options = options

    def calibrate_model(
            self,
            model: OptionPricingModel,
            method=None,
            options=None,
            bounds='default'
    ) -> Tuple[OptimizeResult, OptionPricingModel]:
        """
        Tune model parameters and returns a tuned model. The algorithm tries to minimize the squared difference
        between the prices of listed options and the prices predicted by the model, by tuning model parameters.
        The numerical optimization is performed by :func:`~scipy.optimize.minimize` in the scipy package.

        :param model: the model to calibrate
        :param method: see :func:`~scipy.optimize.minimize`
        :param options: see :func:`~scipy.optimize.minimize`
        :param bounds: the bounds to apply to parameters. If none is specified, then the
                       :attr:`~DEFAULT_PARAMETER_LOWER_BOUND` is applied for all the parameters.
                       Otherwise, a list of tuples (lower_bound, upper_bound) for each parameter shall be specified.
        :return: a tuple (res, model), where res is the result of :func:`~scipy.optimize.minimize`,
                 while model a calibrated model
        """
        if bounds == 'default':
            bounds = ((self.DEFAULT_PARAMETER_LOWER_BOUND, None),) * len(model.parameters)
        loss = self._get_loss_function(model)
        res = minimize(loss, np.array(model.parameters), bounds=bounds, method=method, options=options)
        model.parameters = res.x
        return res, model

    def _get_loss_function(self, model):
        def _loss_function(parameters):
            model.parameters = parameters
            predicted_prices = np.array([model.price_option_black(o) for o in self.options])
            real_prices = np.array([o.price for o in self.options])
            return ((predicted_prices - real_prices) ** 2).sum()

        return _loss_function
