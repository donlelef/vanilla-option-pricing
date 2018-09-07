import numpy as np
from py_vollib.black import undiscounted_black
from py_vollib.black_scholes_merton import black_scholes_merton


class OptionPricingModel:

    def __init__(self, model):
        self.model = model

    @property
    def parameters(self):
        return self.model.parameters

    @parameters.setter
    def parameters(self, value):
        self.model.parameters = value

    def variance(self, t):
        return self.model.variance(t)

    def standard_deviation(self, t):
        return np.sqrt(self.variance(t))

    def volatility(self, t):
        return self.standard_deviation(t) / np.sqrt(t)

    def price_black_scholes_merton(self, option_type, spot, strike, years_to_maturity, risk_free_rate, dividend=0):
        volatility = self.volatility(years_to_maturity)
        price = black_scholes_merton(option_type, spot, strike, years_to_maturity, risk_free_rate, volatility, dividend)
        return price

    def price_option_black_scholes_merton(self, option, risk_free_rate: float):
        return self.price_black_scholes_merton(
            option.option_type,
            option.spot,
            option.strike,
            option.years_to_maturity,
            risk_free_rate,
            option.dividend
        )

    def price_black(self, option_type, spot, strike, years_to_maturity):
        volatility = self.volatility(years_to_maturity)
        price = undiscounted_black(spot, strike, volatility, years_to_maturity, option_type)
        return price

    def price_option_black(self, option):
        return self.price_black(option.option_type, option.spot, option.strike, option.years_to_maturity)
