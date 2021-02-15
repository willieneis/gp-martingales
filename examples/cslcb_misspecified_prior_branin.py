from argparse import Namespace
import copy
import pickle
import numpy as np

from src.simple_gp import SimpleGp
import src.gp_util as gput
from src.quotient import normal_quotient, normal_quotient_log_z
from src.branin import branin, get_branin_domain


seed = 1
np.random.seed(seed)

def run_cslcb():
    """Run LCB algorithm with confidence sequence (CS-LCB)."""

    # Define model params
    gp2_hypers = {'ls': 7., 'alpha': 0.1, 'sigma': 1e-1}

    # Define domain
    x_min = -10.
    x_max = 10.
    domain = get_branin_domain()

    # Setup BO: initialize data
    data = Namespace(X=[], y=np.array([]))
    init_x = domain.unif_rand_sample()
    data = query_function_update_data(init_x, data, branin)

    n_iter = 50
    n_test_pts = 100

    print('Finished iter: ', end='')
    for i in range(n_iter):
        
        # Define gp2 (posterior)
        gp2 = SimpleGp(gp2_hypers)
        gp2.set_data(data)

        # Define gp3 (modified prior)
        gp3_hypers = copy.deepcopy(gp2_hypers)
        gp3_hypers['alpha'] += 0.01
        gp3 = SimpleGp(gp3_hypers)

        # Compute lb_list
        test_pts = domain.unif_rand_sample(n_test_pts)
        lb_list, ub_list = get_lb_ub_lists(test_pts, gp2, gp3, data, False)
        min_idx = np.nanargmin(lb_list)
        next_query_point = test_pts[min_idx]
        
        # update data
        data = query_function_update_data(next_query_point, data, branin)
        print('{}, '.format(i), end='')
        
    print('Data:')
    print(data)

    print('-----')
    opt_idx = np.argmin(data.y)
    print('Minimum y = {}'.format(data.y[opt_idx]))
    print('Minimizer x = {}'.format(data.X[opt_idx]))
    print('At iteration {}'.format(opt_idx + 1))
    print('-----')

    return data


def query_function_update_data(query_point, data, f):
    """Query f at query_point. Return updated data."""
    y = f(np.array(query_point))
    data_new = copy.deepcopy(data)
    data_new.X.append(np.array([query_point]).reshape(-1))
    data_new.y = np.append(data_new.y, y)
    return data_new


def get_lb_ub_lists(dom_pt_list, gp_num, gp_den, data, print_pt=True):
    """Return lists of lower bounds and upper bounds given list of domain points."""

    lb_list, ub_list = [], []

    for x in dom_pt_list:
        if print_pt:
            print('domain point x: {}'.format(x))

        x_list = data.X + [np.array(x).reshape(-1)]
        
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
data = run_cslcb()
filename = 'result_cslcb_seed_' + str(seed) + '.pkl'
pickle.dump(data, open(filename, 'wb'))
print('Saved: ' + filename)
