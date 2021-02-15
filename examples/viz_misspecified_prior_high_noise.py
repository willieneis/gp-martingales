from argparse import Namespace
import copy
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from src.simple_gp import SimpleGp
from src.domains import RealDomain
import src.gp_util as gput
from src.quotient import normal_quotient, normal_quotient_log_z
import src.gp_viz as gpv

import neatplot
neatplot.set_style('fonts')


np.random.seed(11)


def make_viz():
    """Produce visualization."""

    # Set plot settings
    ylim = [-5, 5]
    gpv.update_rc_params()

    # Define model params
    gp1_hypers = {'ls': 1., 'alpha': 1.5, 'sigma': 1e-1}
    gp2_hypers = {'ls': 3., 'alpha': 1., 'sigma': 1e-1}

    # Define domain
    x_min = -10.
    x_max = 10.
    domain = RealDomain({'min_max': [(x_min, x_max)]})

    # Drawn one sample path from gp1
    gp1 = SimpleGp(gp1_hypers)
    sample_path = gput.get_sample_path(gp1, domain)

    n_obs_list = [3, 5, 10, 20, 30, 60]

    # Make full data
    data_full = gput.get_data_from_sample_path(
        sample_path, gp1.params.sigma*4.0, 100
    )

    # Define set of domain points
    dom_pt_list = list(np.linspace(x_min, x_max, 200))

    for n_obs in n_obs_list:

        data = get_data_subset(data_full, n_obs)

        # Define gp2 (posterior)
        gp2 = SimpleGp(gp2_hypers)
        gp2.set_data(data)

        # Define gp3 (modified prior)
        gp3_hypers = copy.deepcopy(gp2_hypers)
        gp3_hypers['alpha'] += 0.01
        gp3 = SimpleGp(gp3_hypers)

        lb_list, ub_list = get_lb_ub_lists(dom_pt_list, gp2, gp3, data, False)

        # Various plotting
        plt.figure()

        # Plot C_t bounds
        plt.plot(dom_pt_list, lb_list, 'g--')
        plt.plot(dom_pt_list, ub_list, 'g--')

        plt.fill_between(dom_pt_list, lb_list, ub_list, color='wheat', alpha=0.5)

        # Plot GP posterior
        save_str = 'viz_' + str(n_obs)
        gpv.visualize_gp_and_sample_path(gp2, domain, data, sample_path,
                                         show_sp=False, ylim=ylim,
                                         save_str=save_str)
        plt.close()


def get_data_subset(data_all, n_sub):
    """Return data namespace with first n_sub elements of data.X and data.y."""
    data = copy.deepcopy(data_all)
    data.X = data.X[:n_sub]
    data.y = data.y[:n_sub]
    return data


def get_lb_ub_lists(dom_pt_list, gp_num, gp_den, data, print_pt=True):
    """Return lists of lower bounds and upper bounds given list of domain points."""

    lb_list, ub_list = [], []

    for x in dom_pt_list:
        if print_pt:
            print('domain point x: {}'.format(x))

        x_list = data.X + [np.array([x])]
        
        mu, cov = gp_num.get_gp_post_mu_cov(x_list)
        gp_num_params = {'mean': mu, 'cov': cov}

        mu, cov = gp_den.get_gp_post_mu_cov(x_list)
        gp_den_params = {'mean': mu, 'cov': cov}

        lb, ub = get_conf_bounds(gp_num_params, gp_den_params)

        lb_list.append(lb)
        ub_list.append(ub)

    return lb_list, ub_list


def get_conf_bounds(gp_num_params, gp_den_params):
    """Return lower bound and upper bound."""
    alpha_level = .05

    quot_mean, quot_cov = normal_quotient(gp_num_params['mean'], gp_num_params['cov'],
                                          gp_den_params['mean'], gp_den_params['cov'])

    log_z = normal_quotient_log_z(gp_num_params['mean'], gp_num_params['cov'],
                                  gp_den_params['mean'], gp_den_params['cov'])

    log_quot_at_mean = gput.get_log_mvn_pdf_at_mean(quot_mean, quot_cov)
    log_alpha = np.log(alpha_level)

    m_dist = np.sqrt(2 * (log_quot_at_mean - log_z - log_alpha))

    # Constrain Mahalanobis distance
    m_dist_min = .01
    m_dist_max = 10.
    m_dist = m_dist_min if m_dist < m_dist_min else m_dist
    m_dist = m_dist_max if m_dist > m_dist_max else m_dist

    max_projection = np.sqrt(quot_cov[-1, -1]) * m_dist

    lb = quot_mean[-1] - max_projection
    ub = quot_mean[-1] + max_projection

    return lb, ub


# Script
make_viz()
