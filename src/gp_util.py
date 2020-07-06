"""
Utilities for Gaussian processes.
"""

from argparse import Namespace
from copy import deepcopy
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvnorm
from sklearn.gaussian_process.kernels import Matern


def get_sample_path(gp, domain, n_grid=500):
    """
    Return a single sample path from the GP. Sample path is a Namespace with
    fields x and y, both 1D numpy arrays.
    """
    # Select test_pts as a grid in the domain (using first dimension only)
    test_pts = np.linspace(
        domain.params.min_max[0][0], domain.params.min_max[0][1], n_grid
    )
    test_pts_list = [np.array([tp]) for tp in test_pts]

    # Compute list of samples from posterior predictive (for each test point)
    sample_list = gp.sample_gp_post(test_pts_list, 1)

    # Compute y
    y = np.array([sample[0] for sample in sample_list])

    # Construct sample_path Namespace
    sample_path = Namespace
    sample_path.x = test_pts
    sample_path.y = y

    return sample_path

def get_data_from_sample_path(sample_path, gp_noise_var, n_obs=3):
    """Draw a few observations from sample_path."""
    obs_idx_list = np.random.choice(range(len(sample_path.x)), n_obs)
    data = Namespace()
    data.X = [np.array([sample_path.x[idx]]) +
              np.random.normal(scale=gp_noise_var)
              for idx in obs_idx_list]
    data.y = np.array([sample_path.y[idx] for idx in obs_idx_list])
    return data

def kern_exp_quad(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel).
    """
    return alpha**2 * kern_exp_quad_noscale(xmat1, xmat2, ls)

def kern_exp_quad_noscale(xmat1, xmat2, ls):
    """
    Exponentiated quadratic kernel function (aka squared exponential kernel aka
    RBF kernel), without scale parameter.
    """
    sq_norm = (-1/(2 * ls**2)) * cdist(xmat1, xmat2, 'sqeuclidean')
    return np.exp(sq_norm)

def squared_euc_distmat(xmat1, xmat2, coef=1.):
    """
    Distance matrix of squared euclidean distance (multiplied by coef) between
    points in xmat1 and xmat2.
    """
    return coef * cdist(xmat1, xmat2, 'sqeuclidean')

def kern_distmat(xmat1, xmat2, ls, alpha, distfn):
    """
    Kernel for a given distmat, via passed in distfn (which is assumed to be fn
    of xmat1 and xmat2 only).
    """
    distmat = distfn(xmat1, xmat2)
    sq_norm = -distmat / ls**2
    return alpha**2 * np.exp(sq_norm)

def kern_matern(xmat1, xmat2, ls, alpha, nu=1.):
    """Matern kernel."""
    xmat1 = np.array(xmat1)
    xmat2 = np.array(xmat2)
    kern = alpha * Matern(length_scale=ls, nu=nu, length_scale_bounds='fixed')
    return kern(xmat1, xmat2)

def get_cholesky_decomp(k11_nonoise, sigma, psd_str):
    """Return cholesky decomposition."""
    if psd_str == 'try_first':
        k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
        try:
            return stable_cholesky(k11, False)
        except np.linalg.linalg.LinAlgError:
            return get_cholesky_decomp(k11_nonoise, sigma, 'project_first')
    elif psd_str == 'project_first':
        k11_nonoise = project_symmetric_to_psd_cone(k11_nonoise)
        return get_cholesky_decomp(k11_nonoise, sigma, 'is_psd')
    elif psd_str == 'is_psd':
        k11 = k11_nonoise + sigma**2 * np.eye(k11_nonoise.shape[0])
        return stable_cholesky(k11)

def stable_cholesky(mmat, make_psd=True):
    """Return a 'stable' cholesky decomposition of mmat."""
    if mmat.size == 0:
        return mmat
    try:
        lmat = np.linalg.cholesky(mmat)
    except np.linalg.linalg.LinAlgError as e:
        if not make_psd:
            raise e
        diag_noise_power = -11
        max_mmat = np.diag(mmat).max()
        diag_noise = np.diag(mmat).max() * 1e-11
        break_loop = False
        while not break_loop:
            try:
                lmat = np.linalg.cholesky(mmat + ((10**diag_noise_power) *
                                                  max_mmat) *
                                                  np.eye(mmat.shape[0]))
                break_loop = True
            except np.linalg.linalg.LinAlgError:
                if diag_noise_power > -9:
                    print('\tstable_cholesky failed with '
                          'diag_noise_power=%d.'%(diag_noise_power))
                diag_noise_power += 1
            if diag_noise_power >= 5:
                print('\t***** stable_cholesky failed: added diag noise '
                      '= %e'%(diag_noise))
    return lmat

def project_symmetric_to_psd_cone(mmat, is_symmetric=True, epsilon=0):
    """Project symmetric matrix mmat to the PSD cone."""
    if is_symmetric:
        try:
            eigvals, eigvecs = np.linalg.eigh(mmat)
        except np.linalg.LinAlgError:
            print('\tLinAlgError encountered with np.eigh. Defaulting to eig.')
            eigvals, eigvecs = np.linalg.eig(mmat)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
    else:
        eigvals, eigvecs = np.linalg.eig(mmat)
    clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
    return (eigvecs * clipped_eigvals).dot(eigvecs.T)

def solve_lower_triangular(amat, b):
    """Solves amat*x=b when amat is lower triangular."""
    return solve_triangular_base(amat, b, lower=True)

def solve_upper_triangular(amat, b):
    """Solves amat*x=b when amat is upper triangular."""
    return solve_triangular_base(amat, b, lower=False)

def solve_triangular_base(amat, b, lower):
    """Solves amat*x=b when amat is a triangular matrix."""
    if amat.size == 0 and b.shape[0] == 0:
        return np.zeros((b.shape))
    else:
        return solve_triangular(amat, b, lower=lower)

def sample_mvn(mu, covmat, nsamp):
    """
    Sample from multivariate normal distribution with mean mu and covariance
    matrix covmat.
    """
    mu = mu.reshape(-1,)
    ndim = len(mu)
    lmat = stable_cholesky(covmat)
    umat = np.random.normal(size=(ndim, nsamp))
    return lmat.dot(umat).T + mu

def gp_post(x_train, y_train, x_pred, ls, alpha, sigma, kernel, full_cov=True):
    """Compute parameters of GP posterior."""
    k11_nonoise = kernel(x_train, x_train, ls, alpha)
    lmat = get_cholesky_decomp(k11_nonoise, sigma, 'try_first')
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y_train))
    k21 = kernel(x_pred, x_train, ls, alpha)
    mu2 = k21.dot(smat)
    k22 = kernel(x_pred, x_pred, ls, alpha)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T.dot(vmat)
    if full_cov is False:
        k2 = np.sqrt(np.diag(k2))
    return mu2, k2

def get_gp_prior_params(x_pred, ls, alpha, kernel, full_cov=True):
    """Compute parameters of GP prior. Assumes zero mean."""
    cov = kernel(x_pred, x_pred, ls, alpha)
    mu = np.zeros(len(x_pred))
    return mu, cov

def get_pdf_ratio(s, prior_mean, prior_cov, post_mean, post_cov,
                  allow_singular=False):
    """Return prior over posterior density ratio for s."""
    prior_pdf = mvnorm.pdf(s, prior_mean, prior_cov,
                           allow_singular=allow_singular)
    post_pdf = mvnorm.pdf(s, post_mean, post_cov,
                          allow_singular=allow_singular)
    pdf_ratio = prior_pdf / post_pdf
    return pdf_ratio

def get_log_pdf_ratio(s, prior_mean, prior_cov, post_mean, post_cov,
                      allow_singular=False):
    """Return log of prior over posterior density ratio for s."""
    # Using scipy multivariate_normal
    log_prior_pdf = mvnorm.logpdf(s, prior_mean, prior_cov,
                                  allow_singular=allow_singular)
    log_post_pdf = mvnorm.logpdf(s, post_mean, post_cov,
                                 allow_singular=allow_singular)
    log_pdf_ratio = log_prior_pdf - log_post_pdf
    return log_pdf_ratio

def get_quot_log_z(gp_num_mean, gp_num_cov, gp_den_mean, gp_den_cov):
    """Return log of normalizing constant for Gaussian quotient."""
    cov_diff = gp_den_cov - gp_num_cov

    (sign, logdet) = np.linalg.slogdet(cov_diff)
    #t1 = sign * np.exp(logdet)
    t1 = logdet

    (sign, logdet) = np.linalg.slogdet(gp_den_cov)
    #t2 = sign * np.exp(logdet)
    t2 = logdet

    #t3 = mvnorm.pdf(gp_num_mean, gp_den_mean, cov_diff, allow_singular=True)
    t3 = mvnorm.logpdf(gp_num_mean, gp_den_mean, cov_diff, allow_singular=True)

    #z = t1 * t3 / t2
    z = t1 + t3 - t2
    return z

def get_log_mvn_pdf_at_mean(mvn_mean, mvn_cov):
    """Return log of mvn pdf evaluated at its mean."""

    try:
        log_pdf = mvnorm.logpdf(mvn_mean, mvn_mean, mvn_cov, allow_singular=True)
    except:
        print('In get_log_mvn_pdf_at_mean, mvn_cov not PSD. Making approximation')
        mvn_cov = near_psd(mvn_cov)
        log_pdf = mvnorm.logpdf(mvn_mean, mvn_mean, mvn_cov, allow_singular=True)

    return log_pdf

def near_psd(A, epsilon=0):
    """Credit: https://stackoverflow.com/a/18542094"""
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return(out)

def get_log_mvn_pdf(eval_point, mvn_mean, mvn_cov):
    """Return log of mvn pdf evaluated at eval_point."""
    log_pdf = mvnorm.logpdf(eval_point, mvn_mean, mvn_cov, allow_singular=True)
    return log_pdf

def get_mahalanobis_dist(x, mu, cov):
    """Return Mahalanobis distance."""
    mean_diff = x - mu
    mean_diff.reshape(-1, 1)
    inv_prod = np.linalg.solve(cov, mean_diff)
    m_squared = np.matmul(mean_diff.T, inv_prod)
    m = np.sqrt(m_squared)
    return m
