from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from src.simple_gp import SimpleGp
from src.domains import RealDomain
import src.gp_viz as gpv


np.random.seed(12)


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

    # Define gp1 and gp2
    gp1 = SimpleGp(gp1_hypers)
    gp2 = SimpleGp(gp2_hypers)

    # Plot GP priors
    data = Namespace(X=[], y=np.array([]))
    gpv.visualize_gp(gp1, domain, data, std_mult=2, ylim=ylim, save_str='gp_prior_1')
    plt.close()
    gpv.visualize_gp(gp2, domain, data, std_mult=2, ylim=ylim, save_str='gp_prior_2')
    plt.close()


# Script
make_viz()
