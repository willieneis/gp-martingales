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


np.random.seed(13)

def run_cslcb():
    """Run LCB algorithm with confidence sequence (CS-LCB)."""

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
    sample_path_nonoise = gput.get_sample_path(gp1, domain)

    # Convert to noisy sample_path observations
    sample_path = gput.get_noisy_sample_path(sample_path_nonoise, gp1.params.sigma)

    # Setup BO: initialize data
    data = Namespace(X=[], y=np.array([]))
    init_x = np.random.choice(sample_path.x, 5)[2]
    data = query_function_update_data(init_x, data, sample_path)

    # Define set of domain points
    dom_pt_list = list(sample_path.x) # NOTE: confirm if correct

    n_iter = 20

    print('Finished iter: ', end='')
    for i in range(n_iter):
        
        # Define gp2 (posterior)
        gp2 = SimpleGp(gp2_hypers)
        gp2.set_data(data)

        # Define gp3 (modified prior)
        gp3_hypers = copy.deepcopy(gp2_hypers)
        gp3_hypers['alpha'] += 0.01 # TODO: define somewhere else?
        gp3 = SimpleGp(gp3_hypers)

        lb_list, ub_list = get_lb_ub_lists(dom_pt_list, gp2, gp3, data, False)
        min_idx = np.nanargmin(lb_list)
        next_query_point = dom_pt_list[min_idx]
        
        # Various plotting
        plt.figure()

        # Plot C_t bounds
        plt.plot(dom_pt_list, lb_list, 'g--')
        plt.plot(dom_pt_list, ub_list, 'g--')

        plt.fill_between(dom_pt_list, lb_list, ub_list, color='wheat', alpha=0.5)

        # Plot GP posterior
        save_str = 'viz_' + str(i)
        gpv.visualize_sample_path_and_data(sample_path_nonoise, data, ylim=ylim,
                                           save_str=save_str)
        plt.close()
        
        # update data
        data = query_function_update_data(next_query_point, data, sample_path)
        print('{}, '.format(i), end='')
        

    print('Data:')
    print(data)


def query_function_update_data(query_point, data, sample_path):
    """Assume query_point is in sample_path.x. Return updated data."""
    data_new = copy.deepcopy(data)
    query_idx = np.where(sample_path.x == query_point)[0][0]
    data_new.X.append(np.array([query_point]))
    data_new.y = np.append(data_new.y, sample_path.y[query_idx])
    return data_new


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
run_cslcb()
