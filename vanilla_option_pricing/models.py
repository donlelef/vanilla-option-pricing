import abc
from typing import Tuple

import numpy as np
from scipy import linalg as la

from vanilla_option_pricing.option_pricing import OptionPricingModel


class PossiblePricingModel(abc.ABC):
    """
    A model which can be used to price options, because it exposes the methods
    required by :class:`~option_pricing.OptionPricingModel`.
    """

    def as_option_pricing_model(self) -> OptionPricingModel:
        """
        Converts the model to as option pricing model, thus providing pricing methods

        :return: an :class:`~option_pricing.OptionPricingModel` based on this model
        """
        return OptionPricingModel(self)


class LogMeanRevertingToGeneralisedWienerProcess(PossiblePricingModel):
    """
    The Log Mean-Reverting To Generalised Wiener Process model. It is a two-factor, mean reverting model, where the
    long-term behaviour is given by a Geometric Brownian motion, while the short-term mean-reverting tendency
    is modelled by an Ornstein-Uhlenbeck process.

    :param p_0: the initial variance, that is the variance of the state at time t=0. Must be a 2x2 numpy array
    :param l: the strength of mean-reversion
    :param s_x: volatility of the long-term process
    :param s_y: volatility of the short-term process
    """

    name = 'Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.array, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.l = l
        self.s_x = s_x
        self.s_y = s_y

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """
        Model parameters, as a list of real numbers, in the order [l, s_x, s_y].
        """
        return self.l, self.s_x, self.s_y

    @parameters.setter
    def parameters(self, value: Tuple[float, float, float]):
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        first_term = (self.p_0[0, 0] - 2 * self.p_0[1, 0] + self.p_0[1, 1] - (self.s_x ** 2 + self.s_y ** 2) / (
                2 * self.l)) * np.exp(-2 * self.l * t)
        second_term = 2 * (self.p_0[1, 0] - self.p_0[1, 1] + self.s_y ** 2 / self.l) * np.exp(-self.l * t)
        third_term = (self.s_y ** 2) * t + (self.s_x ** 2 - 3 * (self.s_y ** 2)) / (2 * self.l) + self.p_0[1, 1]
        return first_term + second_term + third_term


class OrnsteinUhlenbeck(PossiblePricingModel):
    """
    The single-factor, mean-reverting Ornstein-Uhlenbeck process.

    :param p_0: the initial variance, that is the variance of the state at time t=0
    :param l: the strength of the mean-reversion
    :param s: the volatility
    """
    name = 'Ornstein-Uhlenbeck'

    def __init__(self, p_0: float, l: float, s: float):
        self.p_0 = p_0
        self.l = l
        self.s = s

    @property
    def parameters(self) -> Tuple[float, float]:
        """
        Model parameters, as a list of real numbers, in the order [l, s].
        """
        return self.l, self.s

    @parameters.setter
    def parameters(self, value: Tuple[float, float]):
        self.l = value[0]
        self.s = value[1]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.p_0 * np.exp(-2 * self.l * t) + self.s ** 2 / (2 * self.l) * (1 - np.exp(-2 * self.l * t))


class BlackScholes(PossiblePricingModel):
    """
    The famous Black-Sholes model, basically a Geometric Brownian Motion

    :param s: the volatility
    """
    name = 'Black-Sholes'

    def __init__(self, s: float):
        self.s = s

    @property
    def parameters(self) -> Tuple[float]:
        """
        Model parameters, as a list of real numbers, in the order [s].
        """
        return (self.s,)

    @parameters.setter
    def parameters(self, value: Tuple[float]):
        self.s = value[0]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.s ** 2 * t


class NumericalLogMeanRevertingToGeneralisedWienerProcess(PossiblePricingModel):
    """
    This model relies on the same stochastic process as :class:`~models.LogMeanRevertingToGeneralisedWienerProcess`,
    but uses numerical procedures based on matrix exponential instead of closed formulas to compute the variance.
    As this approach is considerably slower, it is strongly suggested to use
    :class:`~models.LogMeanRevertingToGeneralisedWienerProcess` instead, using this class only for benchmarking

    :param p_0: the initial variance, that is the variance of the state at time t=0. Must be a 2x2 numpy matrix
    :param l: the strength of mean-reversion
    :param s_x: volatility of the long-term process
    :param s_y: volatility of the short-term process
    """

    name = 'Numerical Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.array, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.l = l
        self.s_x = s_x
        self.s_y = s_y
        self.numerical_model = NumericalModel(self.__get_A_matrix(), self.__get_B_matrix(), self.p_0)

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """
        Model parameters, as a list of real numbers, in the order [l, s_x, s_y].
        """
        return self.l, self.s_x, self.s_y

    @parameters.setter
    def parameters(self, value: Tuple[float, float, float]):
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]
        self.numerical_model.parameters = (self.__get_A_matrix(), self.__get_B_matrix())

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.numerical_model.variance(t)

    def __get_A_matrix(self) -> np.array:
        return np.array([[-self.l, self.l], [0, 0]])

    def __get_B_matrix(self) -> np.array:
        return np.array([[self.s_x, 0], [0, self.s_y]])


class NumericalModel:
    """
    A general-purpose linear stochastic system. All the parameters must be matrices of suitable dimension

    :param A: the dynamic matrix A of the system
    :param B: the input matrix B of the system
    :param p_0: the initial variance, that is the variance of the state at time t=0

    """

    def __init__(self, A: np.array, B: np.array, p_0: np.array):
        self.A = A
        self.B = B
        self.p_0 = p_0

    @property
    def parameters(self) -> Tuple[np.array, np.array]:
        """
        Model parameters, as a list of matrices, in the order [A, B].
        """
        return self.A, self.B

    @parameters.setter
    def parameters(self, value: Tuple[np.array, np.array]):
        self.A = value[0]
        self.B = value[0]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        dim = self.A.shape[0]
        F = la.expm(np.block([
            [self.A, self.B @ np.transpose(self.B)],
            [np.zeros_like(self.A), -np.transpose(self.A)]
        ]) * t)
        P = (F[0:dim, 0:dim] @ self.p_0 + F[0:dim, dim:2 * dim]) @ la.inv(F[dim:2 * dim, dim:2 * dim])
        return P[0, 0]
