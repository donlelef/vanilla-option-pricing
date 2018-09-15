import pandas as pd
from py_lets_be_rational.exceptions import BelowIntrinsicException
from py_vollib.black import implied_volatility as iv


def option_list_to_pandas_dataframe(options):
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
    return [VanillaOption(**o) for o in data_frame.to_dict(orient='record')]


class VanillaOption:
    DAYS_IN_YEAR = 365.2425

    def __init__(self, instrument: str, option_type, date, price: float, strike: float, spot: float,
                 maturity, dividend=0):
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
        return (self.maturity - self.date).days / self.DAYS_IN_YEAR

    @property
    def implied_volatility_of_undiscounted_price(self):
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
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}
