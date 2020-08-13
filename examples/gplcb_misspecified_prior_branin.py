from argparse import Namespace
import copy
import pickle
import numpy as np

from src.simple_gp import SimpleGp
import src.gp_util as gput
from src.branin import branin, get_branin_domain


seed = 1
np.random.seed(seed)

def run_gplcb():
    """Run LCB algorithm with Gaussian process (GP-LCB)."""

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

        # Compute lb_list
        test_pts = domain.unif_rand_sample(n_test_pts)
        mean_list, std_list = gp2.get_gp_post_mu_cov(test_pts, full_cov=False)
        conf_mean = np.array(mean_list)
        std_mult = 3.
        lb_list = conf_mean - std_mult * np.array(std_list) # technically is np ndarray
        ub_list = conf_mean + std_mult * np.array(std_list) # technically is np ndarray

        # Compute next_query_point
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


# Script
data = run_gplcb()
filename = 'result_gplcb_seed_' + str(seed) + '.pkl'
pickle.dump(data, open(filename, 'wb'))
print('Saved: ' + filename)
