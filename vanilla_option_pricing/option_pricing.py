from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np
from py_vollib.black import undiscounted_black
from py_vollib.black_scholes_merton import black_scholes_merton

from vanilla_option_pricing.option import VanillaOption, check_option_type


class OptionPricingModel(ABC):
    """
    A model which can be used to price European vanilla options.
    """

    @property
    @abstractmethod
    def parameters(self) -> Sequence[float]:
        """
        The model parameters, returned as a list of real numbers.
        """
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, value: Sequence[float]):
        """
        The model parameters, returned as a list of real numbers.
        """
        pass

    @abstractmethod
    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time.

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        pass

    def standard_deviation(self, t: float) -> float:
        """
        The standard deviation of the model output at a given instant,
        that is the squared root of the :func:`~option_pricing.OptionPricingModel.variance`
        at the same instant

        :param t: the time when the standard deviation is evaluated
        :return: the standard deviation at time t
        """
        return np.sqrt(self.variance(t))

    def volatility(self, t: float) -> float:
        """
        The volatility of the model output at a certain instant,
        that is the :func:`~option_pricing.OptionPricingModel.standard_deviation`
        divided by the squared root of the time

        :param t: the time when the volatility is evaluated
        :return: the volatility at time t
        """
        return self.standard_deviation(t) / np.sqrt(t)

    def price_black_scholes_merton(self, option_type: str, spot: float, strike: float, years_to_maturity: float,
                                   risk_free_rate: float, dividend: float = 0) -> float:
        """
        Finds the no-arbitrage price of a European Vanilla option. The price is computed using the Black-Scholes-Merton
        framework, but the variance of the underlying is extracted from this model.

        :param option_type: the type of the option (c for call, p for put)
        :param spot: the spot price of the underlying
        :param strike: the option strike price
        :param years_to_maturity: the years remaining before maturity - as a decimal number
        :param risk_free_rate: the risk-free interest rate
        :param dividend: the dividend paid by the underlying - as a decimal number
        :return: the no-arbitrage price of the option
        """
        volatility = self.volatility(years_to_maturity)
        check_option_type(option_type)
        price = black_scholes_merton(option_type, spot, strike, years_to_maturity, risk_free_rate, volatility, dividend)
        return price

    def price_option_black_scholes_merton(self, option: VanillaOption, risk_free_rate: float) -> float:
        """
        Same as :func:`~option_pricing.OptionPricingModel.price_black_scholes_merton`, but the details
        of the vanilla option are provided by a :class:`~option.VanillaOption` object.

        :param option: a :class:`~option.VanillaOption`
        :param risk_free_rate: the risk-free interest rate
        :return: the no-arbitrage price of the option
        """
        return self.price_black_scholes_merton(
            option.option_type,
            option.spot,
            option.strike,
            option.years_to_maturity,
            risk_free_rate,
            option.dividend
        )

    def price_black(self, option_type: str, spot: float, strike: float, years_to_maturity: float) -> float:
        """
        Finds the no-arbitrage price of a European Vanilla option. Price is computed using the Black
        formulae, but the variance of the underlying is extracted from this model.

        :param option_type: the type of the option (c for call, p for put)
        :param spot: the spot price of the underlying
        :param strike: the option strike price
        :param years_to_maturity: the years remaining before maturity - as a decimal number
        :return: the no-arbitrage price of the option
        """
        volatility = self.volatility(years_to_maturity)
        check_option_type(option_type)
        price = undiscounted_black(spot, strike, volatility, years_to_maturity, option_type)
        return price

    def price_option_black(self, option: VanillaOption) -> float:
        """
        Same as :func:`~option_pricing.OptionPricingModel.price_black`, but the details
        of the vanilla option are provided by a :class:`~option.VanillaOption` object.

        :param option: a :class:`~option.VanillaOption`
        :return: the no-arbitrage price of the option
        """
        return self.price_black(option.option_type, option.spot, option.strike, option.years_to_maturity)

    @staticmethod
    def _check_positivity(params: Iterable[float], message=''):
        if any(x < 0 for x in params):
            raise ValueError('All values must be non-negative. ' + message)
