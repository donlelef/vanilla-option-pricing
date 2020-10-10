from datetime import datetime
from typing import List

import pandas as pd
from py_lets_be_rational.exceptions import BelowIntrinsicException
from py_vollib.black import implied_volatility as iv


class VanillaOption:
    """
    A European vanilla option. All the prices must have consistent unit of measure

    :param instrument: name of the underlying
    :param option_type: type of the option (c for call, p for put)
    :param date: the date when the option is traded
    :param price: option price
    :param strike: option strike price
    :param spot: spot price of the underlying
    :param maturity: the maturity date
    :param dividend: underlying dividend - if any, expressed as a decimal number
    """

    """Number of days in a year"""
    DAYS_IN_YEAR = 365.2425

    def __init__(self, instrument: str, option_type: str, date: datetime, price: float, strike: float, spot: float,
                 maturity: datetime, dividend=0):
        self.instrument = instrument
        self.option_type = option_type.lower()
        self.date = date
        self.price = price
        self.strike = strike
        self.spot = spot
        self.maturity = maturity
        self.dividend = dividend

    @property
    def years_to_maturity(self) -> float:
        """
        The years remaining to option maturity, as a decimal number
        """
        return (self.maturity - self.date).days / self.DAYS_IN_YEAR

    @property
    def implied_volatility_of_undiscounted_price(self) -> float:
        """
        The implied volatility of the option, considering an undiscounted price.
        Returns zero if the computed implied volatility is negative.
        """
        if self.option_type not in ('c', 'p'):
            raise ValueError('option_type shall be either "c" for call or "p" for put')
        try:
            return iv.implied_volatility_of_undiscounted_option_price(
                self.price,
                self.spot,
                self.strike,
                self.years_to_maturity,
                self.option_type
            )
        except BelowIntrinsicException:
            return 0

    def to_dict(self):
        """
        :return: all the fields of the object in a dictionary
        """
        return vars(self)


def option_list_to_pandas_dataframe(options: List[VanillaOption]):
    """
    A utility function to convert a list of :class:`~option.VanillaOption` to a pandas dataframe

    :param options: a list of :class:`~option.VanillaOption`
    :return: a pandas dataframe, containing option data
    """

    return pd.DataFrame.from_records([o.to_dict() for o in options])


def pandas_dataframe_to_option_list(data_frame: pd.DataFrame):
    """
    A utility function to convert a pandas dataframe to a list of :class:`~option.VanillaOption`.
    For this function to work, the dataframe columns should be names as the parameters of
    :class:`~option.VanillaOption`'s constructor

    :param data_frame: a pandas dataframe, containing option data
    :return: a list of :class:`~option.VanillaOption`
    """
    return [VanillaOption(**o) for o in data_frame.to_dict(orient='records')]
