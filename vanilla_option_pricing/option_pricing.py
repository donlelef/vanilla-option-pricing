import numpy as np
from py_vollib.black import undiscounted_black
from py_vollib.black_scholes_merton import black_scholes_merton


class OptionPricingModel:
    """
    A model which can be used to price European vanilla options.

    :param model: the stochastic model of the underlying
    """

    def __init__(self, model):
        self.model = model

    @property
    def parameters(self):
        """
        The model parameters, returned as a list of values
        """
        return self.model.parameters

    @parameters.setter
    def parameters(self, value):
        self.model.parameters = value

    def variance(self, t):
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.model.variance(t)

    def standard_deviation(self, t):
        """
        The standard deviation of the model output at a certain time instant,
        that is the squared root of the :func:`~option_pricing.OptionPricingModel.variance`
        at the same time instant

        :param t: the time when the standard deviation is evaluated
        :return: the standard deviation at time t
        """
        return np.sqrt(self.variance(t))

    def volatility(self, t):
        """
        The volatility of the model output at a certain time instant,
        that is the :func:`~option_pricing.OptionPricingModel.standard_deviation`
        divided by the squared root of the time

        :param t: the time when the volatility is evaluated
        :return: the volatility at time t
        """
        return self.standard_deviation(t) / np.sqrt(t)

    def price_black_scholes_merton(self, option_type, spot, strike, years_to_maturity, risk_free_rate, dividend=0):
        """
        Finds the no-arbitrage price of a European Vanilla option. Price is computed using the Black-Scholes-Merton
        formulae, but the variance of the underlying is extracted from this model.

        :param option_type: the type of the option (c for call, p for put)
        :param spot: the spot price of the underlying
        :param strike: the option strike price
        :param years_to_maturity: the years remaining before maturity - as a decimal number
        :param risk_free_rate: the risk-free interest rate
        :param dividend: the dividend paid by the underlying - as a decimal number
        :return: the no-arbitrage price of the option
        """
        volatility = self.volatility(years_to_maturity)
        price = black_scholes_merton(option_type, spot, strike, years_to_maturity, risk_free_rate, volatility, dividend)
        return price

    def price_option_black_scholes_merton(self, option, risk_free_rate: float):
        """
        Same as :func:`~option_pricing.OptionPricingModel.price_black_scholes_merton`, but the details
        of the vanilla option are provided by a :class:`~option.VanillaOption` object

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

    def price_black(self, option_type, spot, strike, years_to_maturity):
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
        price = undiscounted_black(spot, strike, volatility, years_to_maturity, option_type)
        return price

    def price_option_black(self, option):
        """
        Same as :func:`~option_pricing.OptionPricingModel.price_black`, but the details
        of the vanilla option are provided by a :class:`~option.VanillaOption` object

        :param option: a :class:`~option.VanillaOption`
        :return: the no-arbitrage price of the option
        """
        return self.price_black(option.option_type, option.spot, option.strike, option.years_to_maturity)
