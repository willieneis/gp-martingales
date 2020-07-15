from argparse import Namespace
import copy
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from src.simple_gp import SimpleGp
from src.domains import RealDomain
import src.gp_util as gput
import src.gp_viz as gpv


np.random.seed(13)

def run_gplcb():
    """Run LCB algorithm with Gaussian process (GP-LCB)."""

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

        # Compute lb_list
        test_pts = [np.array([dom_pt]) for dom_pt in dom_pt_list]
        mean_list, std_list = gp2.get_gp_post_mu_cov(test_pts, full_cov=False)
        conf_mean = np.array(mean_list)
        std_mult = 3.
        lb_list = conf_mean - std_mult * np.array(std_list) # technically is np ndarray
        ub_list = conf_mean + std_mult * np.array(std_list) # technically is np ndarray

        # Compute next_query_point
        min_idx = np.nanargmin(lb_list)
        next_query_point = dom_pt_list[min_idx]
        
        # Various plotting
        plt.figure()

        # Plot C_t bounds
        plt.plot(dom_pt_list, mean_list, 'k--', lw=2)
        plt.plot(dom_pt_list, lb_list, 'k--')
        plt.plot(dom_pt_list, ub_list, 'k--')

        plt.fill_between(dom_pt_list, lb_list, ub_list, color='lightsteelblue',
                         alpha=0.5)

        # Plot GP posterior
        save_str = 'viz_' + str(i)
        gpv.visualize_sample_path_and_data(sample_path, data, ylim=ylim,
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


# Script
run_gplcb()
