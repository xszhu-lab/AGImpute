import math
import numpy as np
from scipy.special import gamma


# -----------------
#  Probabilistic statistical model
# -----------------


def ZeroInflatedNegativeBinomail(data, fai, nu, k):
    dk = data + k
    do = data + 1
    if data == 0:
        return fai + (1 - fai) * math.pow(k / (nu + k), k)
    else:
        return (1 - fai) * np.divide(gamma(dk), (gamma(do) * gamma(k))) * math.pow(k / (nu + k), k) * math.pow(
            nu / (k + nu), data)


def ZeroInflatedPossion(data, fai, nu):
    if data == 0:
        return fai + (1 - fai) * math.exp(-nu)
    else:
        return (1 - fai) * np.divide(math.pow(nu, data), math.factorial(data))  # * math.exp(-nu)


def Gaussian(data, mean, cov):
    lambd = (1 / np.sqrt(2 * np.pi)) * np.exp(-(np.power(data - mean, 2)) / (2 * np.power(cov, 2)))
    return lambd


def Possin(k, cov):
    if k < 0:
        return 1
    else:
        try:
            ans = math.exp(200000)
        except OverflowError:
            ans = float('inf')
        return np.power(cov, k) * math.factorial(int(k)) * math.exp(-cov)
