from typing import Tuple

import numpy as np
from scipy import linalg as la

from vanilla_option_pricing.option_pricing import OptionPricingModel


class LogMeanRevertingToGeneralisedWienerProcess(OptionPricingModel):
    """
    The Log Mean-Reverting To Generalised Wiener Process model. It is a two-factor, mean reverting model, where the
    long-term behaviour is given by a Geometric Brownian motion, while the short-term mean-reverting tendency
    is modelled by an Ornstein-Uhlenbeck process.

    :param p_0: the initial variance, that is the variance of the state at time t=0. Must be a 2x2 numpy array
    :param l: the strength of mean-reversion, must be non-negative
    :param s_x: volatility of the long-term process, must be non-negative
    :param s_y: volatility of the short-term process, must be non-negative
    """

    name = 'Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.array, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.parameters = l, s_x, s_y

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """
        Model parameters, as a tuple of real numbers, in the order l, s_x, s_y.
        """
        return self.l, self.s_x, self.s_y

    @parameters.setter
    def parameters(self, value: Tuple[float, float, float]):
        super(LogMeanRevertingToGeneralisedWienerProcess, self)._check_positivity(
            value,
            'l, s_x, s_y must be non-negative'
        )
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        first_term = (self.p_0[0, 0] - 2 * self.p_0[1, 0] + self.p_0[1, 1] - (self.s_x ** 2 + self.s_y ** 2) / (
                2 * self.l)) * np.exp(-2 * self.l * t)
        second_term = 2 * (self.p_0[1, 0] - self.p_0[1, 1] + self.s_y ** 2 / self.l) * np.exp(-self.l * t)
        third_term = (self.s_y ** 2) * t + (self.s_x ** 2 - 3 * (self.s_y ** 2)) / (2 * self.l) + self.p_0[1, 1]
        return first_term + second_term + third_term


class OrnsteinUhlenbeck(OptionPricingModel):
    """
    The single-factor, mean-reverting Ornstein-Uhlenbeck process.

    :param p_0: the initial variance, that is the variance of the state at time t=0, must be positive semidefinite and symmetric
    :param l: the strength of the mean-reversion, must be non-negative
    :param s: the volatility, must be non-negative
    """
    name = 'Ornstein-Uhlenbeck'

    def __init__(self, p_0: float, l: float, s: float):
        self.p_0 = p_0
        self.parameters = l, s

    @property
    def parameters(self) -> Tuple[float, float]:
        """
        Model parameters, as a list of real numbers, in the order [l, s].
        """
        return self.l, self.s

    @parameters.setter
    def parameters(self, value: Tuple[float, float]):
        super(OrnsteinUhlenbeck, self)._check_positivity(
            value,
            'l, s must be non-negative'
        )
        self.l = value[0]
        self.s = value[1]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.p_0 * np.exp(-2 * self.l * t) + self.s ** 2 / (2 * self.l) * (1 - np.exp(-2 * self.l * t))


class GeometricBrownianMotion(OptionPricingModel):
    """
    The celebrated Geometric Brownian Motion model

    :param s: the volatility, must be non-negative
    """
    name = 'Geometric Brownian Motion'

    def __init__(self, s: float):
        self.parameters = (s,)

    @property
    def parameters(self) -> Tuple[float]:
        """
        Model parameters, as a tuple of real numbers, in the order (s, ).
        """
        return (self.s,)

    @parameters.setter
    def parameters(self, value: Tuple[float]):
        super(GeometricBrownianMotion, self)._check_positivity(value, 's must be non-negative')
        self.s = value[0]

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.s ** 2 * t


class NumericalLogMeanRevertingToGeneralisedWienerProcess(OptionPricingModel):
    """
    This model relies on the same stochastic process as :class:`~models.LogMeanRevertingToGeneralisedWienerProcess`,
    but uses a numerical procedures based on a matrix exponential instead of the analytical formulas to
    compute the variance. As this approach is considerably slower, it is strongly suggested to adopt
    :class:`~models.LogMeanRevertingToGeneralisedWienerProcess` instead, using this class only for benchmarking

    :param p_0: the initial variance, that is the variance of the state at time t=0. Must be a 2x2 numpy array, symmetric and positive semidefinite
    :param l: the strength of mean-reversion, must be non-negative
    :param s_x: volatility of the long-term process, must be non-negative
    :param s_y: volatility of the short-term process, must be non-negative
    """

    name = 'Numerical Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.array, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.parameters = l, s_x, s_y

    @property
    def parameters(self) -> Tuple[float, float, float]:
        """
        Model parameters, as a tuple of real numbers, in the order l, s_x, s_y.
        """
        return self.l, self.s_x, self.s_y

    @parameters.setter
    def parameters(self, value: Tuple[float, float, float]):
        super(NumericalLogMeanRevertingToGeneralisedWienerProcess, self)._check_positivity(
            value,
            'l, s_x, s_y must be non-negative'
        )
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]
        self.numerical_model = NumericalModel(self.__get_matrix_a(), self.__get_matrix_b(), self.p_0)

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.numerical_model.variance(t)

    def __get_matrix_a(self) -> np.array:
        return np.array([[-self.l, self.l], [0, 0]])

    def __get_matrix_b(self) -> np.array:
        return np.array([[self.s_x, 0], [0, self.s_y]])


class NumericalModel:
    """
    A general-purpose linear stochastic system. All the parameters must be matrices (as Numpy arrays) of
    suitable dimensions.

    :param A: the dynamic matrix A of the system
    :param B: the input matrix B of the system
    :param p_0: the initial variance, that is the variance of the state at time t=0, must be symmetric and positive semidefinite

    """

    def __init__(self, A: np.array, B: np.array, p_0: np.array):
        self.A = A
        self.B = B
        self.p_0 = p_0

    def variance(self, t: float) -> float:
        """
        The variance of the model output at a given time instant

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
