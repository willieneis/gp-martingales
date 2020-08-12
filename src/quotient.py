"""
Code to compute the quotient of Normal distributions.
"""

from argparse import Namespace
import numpy as np
from scipy.stats import multivariate_normal as mvnorm

from .gp_util import near_psd


def normal_quotient(num_mean, num_cov, den_mean, den_cov):
    """
    Return parameters of normal distribution proportional to:
    N(num_mean, num_cov) / N(den_mean, den_cov).
    """
    # Compute quotient cov
    cov_diff = num_cov - den_cov
    inv_cov_diff_prod = np.linalg.lstsq(cov_diff, num_cov, rcond=None)[0]
    quot_cov = num_cov - np.matmul(num_cov, inv_cov_diff_prod)

    # Compute quotient mean
    inv_num_cov_mean_prod = np.linalg.lstsq(num_cov, num_mean, rcond=None)[0]
    inv_den_cov_mean_prod = np.linalg.lstsq(den_cov, den_mean, rcond=None)[0]
    inv_cov_mean_prod_diff = inv_num_cov_mean_prod - inv_den_cov_mean_prod
    quot_mean = np.matmul(quot_cov, inv_cov_mean_prod_diff)

    # Return normal parameters
    return quot_mean, quot_cov


def normal_quotient_log_z(num_mean, num_cov, den_mean, den_cov):
    """
    Return log of the normalization constant for distribution with density
    proportional to quotient: N(num_mean, num_cov) / N(den_mean, den_cov).
    """
    cov_diff = den_cov - num_cov

    diff_sign, diff_logdet = np.linalg.slogdet(cov_diff)
    den_sign, den_logdet = np.linalg.slogdet(den_cov)

    # To fix numerical issues
    diff_logdet = np.nan_to_num(diff_logdet)
    den_logdet = np.nan_to_num(den_logdet)

    log_det_term =  diff_logdet - den_logdet

    # To fix numerical issues
    if den_logdet < -1e4 or diff_logdet < -1e4:
        log_det_term = 0

    allow_singular = True
    try:
        log_pdf = mvnorm.logpdf(num_mean, den_mean, cov_diff,
                                allow_singular=allow_singular)
    except:
        print(('In normal_quotient_log_z, cov_diff not PSD. ' +
               'Making approximation'))
        cov_diff = near_psd(cov_diff)
        log_pdf = mvnorm.logpdf(num_mean, den_mean, cov_diff,
                                allow_singular=allow_singular)

    # PRINTING
    #print('diff_logdet = {}'.format(diff_logdet))
    #print('den_logdet = {}'.format(den_logdet))
    #print('log_det_term = {}'.format(log_det_term))
    #print('log_pdf = {}'.format(log_pdf))

    log_z = log_det_term + log_pdf

    return log_z
