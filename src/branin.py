"""
Branin synthetic benchmark function.
"""

from argparse import Namespace
import numpy as np

from src.domains import RealDomain


def branin(x):
    """Branin synthetic function wrapper"""

    if len(x.shape) == 0:
        return branin_single(np.array([x]))

    elif len(x.shape) == 1:
        # x must be single input
        return branin_single(x)

    elif len(x.shape) == 2:
        # x could be single input or matrix of multiple inputs
        if x.shape[0] == 1 or x.shape[1] == 1:
            # x is single row matrix or single column matrix
            return branin_single(x.reshape(-1))

        else:
            # Multiple x in a matrix
            return branin_on_matrix(x)

    else:
        raise ValueError(
            (
                'Input to branin function must be a float, or a '
                + '1D or 2D np array, instead of a {}'
            ).format(type(x))
        )


def branin_on_matrix(X):
    """
    Branin synthetic function on matrix of inputs X.

    Parameters
    ----------
    X : ndarray
        A numpy ndarray with shape=(n, ndimx).

    Returns
    -------
    ndarray
        A numpy ndarray with shape=(ndimx,).
    """
    return np.array([branin_single(x) for x in X])


def branin_single(x):
    """
    Branin synthetic function on a single input x.

    Parameters
    ----------
    x : ndarray
        A numpy ndarray with shape=(ndimx,).

    Returns
    -------
    float
        The function value f(x), a float.
    """
    b = 5.1 / (4.0 * np.pi ** 2)
    c = 5.0 / np.pi
    t = 1.0 / (8.0 * np.pi)
    x0 = x[0]
    x1 = x[1]
    return (
        1.0 * (x1 - b * x0 ** 2 + c * x0 - 6.0) ** 2
        + 10.0 * (1 - t) * np.cos(x0)
        + 10.0
    )


def get_branin_domain():
    """Return domain for Branin function"""
    return get_branin_domain_nd(2)


def get_branin_domain_nd(ndimx, verbose=True):
    """Return domain for ndimx-dimensional Branin"""

    if ndimx % 2 == 1:
        raise ValueError('Arg ndimx must be an even number.')

    min_max = [(-5, 10), (0, 15)] * (ndimx // 2)
    domp = Namespace(dom_str='real', min_max=min_max)
    domain = RealDomain(domp, verbose=verbose)

    return domain


def get_branin_product_domain(ndimx, n_prod):
    """Return product domain for ndimx-dimensional Branin."""
    domain_list = get_branin_product_domain_list(ndimx, n_prod)
    domain = ProductDomain(domain_list=domain_list)
    return domain


def get_branin_product_domain_list(ndimx, n_prod):
    """Return ProductDomain domain_list for ndimx-dimensional Branin."""
    domain_list = [get_branin_domain_nd(ndimx // n_prod, False) for _ in range(n_prod)]
    return domain_list
