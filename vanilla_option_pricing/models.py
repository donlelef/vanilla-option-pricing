import abc

import numpy as np
from scipy import linalg as la

from vanilla_option_pricing.option_pricing import OptionPricingModel


class PossiblePricingModel(abc.ABC):
    """
    A model which can be used to price options, because it exposes the methods required by
    :class:`~option_pricing.OptionPricingModel`.
    """
    def as_option_pricing_model(self):
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

    :param p_0: the initial variance. Must be a 2x2 numpy matrix
    :param l: the strength of mean-reversion
    :param s_x: volatility of the long-term process
    :param s_y: volatility of the short-term process
    """

    name = 'Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.matrix, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.l = l
        self.s_x = s_x
        self.s_y = s_y

    @property
    def parameters(self):
        """
        Model parameters, as a list of real numbers, in the order [l, s_x, s_y].
        """
        return [self.l, self.s_x, self.s_y]

    @parameters.setter
    def parameters(self, value):
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]

    def variance(self, t):
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
    name = 'Ornstein-Uhlenbeck'

    def __init__(self, p_0: float, l: float, s: float):
        self.p_0 = p_0
        self.l = l
        self.s = s

    @property
    def parameters(self):
        """
        Model parameters, as a list of real numbers, in the order [l, s].
        """
        return [self.l, self.s]

    @parameters.setter
    def parameters(self, value):
        self.l = value[0]
        self.s = value[1]

    def variance(self, t):
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.p_0 * np.exp(-2 * self.l * t) + self.s ** 2 / (2 * self.l) * (1 - np.exp(-2 * self.l * t))


class BlackScholes(PossiblePricingModel):
    name = 'Black-Sholes'

    def __init__(self, s: float):
        self.s = s

    @property
    def parameters(self):
        """
        Model parameters, as a list of real numbers, in the order [s].
        """
        return [self.s]

    @parameters.setter
    def parameters(self, value):
        self.s = value[0]

    def variance(self, t):
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.s ** 2 * t


class NumericalLogMeanRevertingToGeneralisedWienerProcess(PossiblePricingModel):
    name = 'Numerical Log Mean-Reverting To Generalised Wiener Process'

    def __init__(self, p_0: np.matrix, l: float, s_x: float, s_y: float):
        self.p_0 = p_0
        self.l = l
        self.s_x = s_x
        self.s_y = s_y
        self.numerical_model = NumericalModel(self.__get_A_matrix(), self.__get_B_matrix(), self.p_0)

    @property
    def parameters(self):
        """
        Model parameters, as a list of real numbers, in the order [l, s_x, s_y].
        """
        return [self.l, self.s_x, self.s_y]

    @parameters.setter
    def parameters(self, value):
        self.l = value[0]
        self.s_x = value[1]
        self.s_y = value[2]
        self.numerical_model.parameters = [self.__get_A_matrix(), self.__get_B_matrix()]

    def variance(self, t):
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        return self.numerical_model.variance(t)

    def __get_A_matrix(self):
        return np.matrix([[-self.l, self.l], [0, 0]])

    def __get_B_matrix(self):
        return np.matrix([[self.s_x, 0], [0, self.s_y]])


class NumericalModel:

    def __init__(self, A: np.matrix, B: np.matrix, p_0: np.matrix):
        self.A = A
        self.B = B
        self.p_0 = p_0

    @property
    def parameters(self):
        return [self.A, self.B]

    @parameters.setter
    def parameters(self, value):
        """
        Model parameters, as a list of matrices, in the order [A, B].
        """
        self.A = value[0]
        self.B = value[0]

    def variance(self, t):
        """
        The variance of the model output at a certain time instant

        :param t: the time when the variance is evaluated
        :return: the variance at time t
        """
        dim = self.A.shape[0]
        F = la.expm(
            np.bmat([
                [self.A, self.B * np.transpose(self.B)],
                [np.zeros_like(self.A), -np.transpose(self.A)]
            ]) * t)
        P = (F[0:dim, 0:dim] * self.p_0 + F[0:dim, dim:2 * dim]) * la.inv(F[dim:2 * dim, dim:2 * dim])
        return P[0, 0]
