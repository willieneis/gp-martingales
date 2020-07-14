"""
Code to visualize Gaussian processes.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_gp(gp, domain, data=None, n_grid=500, n_gen=20000, n_samp_plot=10,
                 std_mult=3, show_sp=True, ylim=[-4, 4], save_str=None,
                 exact_post=True):
    """
    Visualize gp posterior predictive, sample paths, and observations on n_grid
    points in 1D domain.
    """
    update_rc_params()

    # Select test_pts as a grid in the domain (using first dimension only)
    test_pts = np.linspace(domain.params.min_max[0][0],
                           domain.params.min_max[0][1], n_grid)
    test_pts_list = [np.array([tp]) for tp in test_pts]

    # Compute list of samples from posterior predictive (for each test point)
    ppred_samp_list = gp.sample_gp_post(test_pts_list, n_gen)

    # Compute and plot sample paths
    if show_sp:
        sampidx = np.random.randint(n_gen, size=(n_samp_plot,))
        samppaths = [np.array([ppred_samp[idx] for ppred_samp in
                               ppred_samp_list])
                     for idx in sampidx]
        for sp in samppaths:
            plt.plot(test_pts, sp)

    # Compute posterior mean and confidence bounds
    if exact_post is True:
        mean_list, std_list = gp.get_gp_post_mu_cov(test_pts_list, full_cov=False)
    else:
        mean_list = [np.mean(ppred_samp) for ppred_samp in ppred_samp_list]
        std_list = [np.std(ppred_samp) for ppred_samp in ppred_samp_list]

    # Plot posterior mean and confidence bounds
    conf_mean = np.array(mean_list)
    conf_bound_lower = conf_mean - std_mult * np.array(std_list)
    conf_bound_upper = conf_mean + std_mult * np.array(std_list)

    plt.plot(test_pts, conf_mean, 'k--', lw=2)
    plt.plot(test_pts, conf_bound_lower, 'k--')
    plt.plot(test_pts, conf_bound_upper, 'k--')

    plt.fill_between(test_pts, conf_bound_lower, conf_bound_upper,
                     color='lightsteelblue', alpha=0.5)

    # Plot data
    if data is not None:
        data_x_arr = np.array(data.X).flatten()
        data_y_arr = data.y
        plt.plot(data_x_arr, data_y_arr, 'o', markeredgecolor='w',
                 markerfacecolor='w', markeredgewidth=2, ms=10)
        plt.plot(data_x_arr, data_y_arr, 'o', markeredgecolor='deeppink',
                 markerfacecolor='deeppink', markeredgewidth=2, ms=6)

    # Plot properties
    xlim = [np.min(test_pts), np.max(test_pts)]
    set_plot_and_show(xlim, ylim)

    # Optionally save figure 
    save_figure(save_str)

def visualize_sample_path(sample_path, ylim=[-4, 4], save_str=None):
    """
    Visualize a sample_path Namespace, assumed to have fields x and y, both 1D
    numpy arrays.
    """
    # Plot sample_path
    plt.plot(sample_path.x, sample_path.y, 'b-', lw=3)

    # Plot properties
    xlim = [np.min(sample_path.x), np.max(sample_path.x)]
    set_plot_and_show(xlim, ylim)

    # Optionally save figure 
    save_figure(save_str)

def visualize_sample_path_and_data(sample_path, data, ylim=[-4, 4],
                                   save_str=None):
    """
    Visualize a sample_path Namespace (assumed to have fields x and y, both 1D
    numpy arrays) and a data Namespace (assumed to have fields X, a list, and
    y, a 1D numpy array).
    """
    update_rc_params()

    # Plot sample_path
    visualize_sample_path(sample_path, ylim)

    # Plot data
    x = np.array([x[0] for x in data.X])
    y = data.y

    data_x_arr = np.array(data.X).flatten()
    data_y_arr = data.y
    plt.plot(data_x_arr, data_y_arr, 'o', markeredgecolor='w',
             markerfacecolor='w', markeredgewidth=2, ms=10)
    plt.plot(data_x_arr, data_y_arr, 'o', markeredgecolor='deeppink',
             markerfacecolor='deeppink', markeredgewidth=2, ms=6)

    xlim = [np.min(sample_path.x), np.max(sample_path.x)]
    set_plot_and_show(xlim, ylim)

    # Optionally save figure 
    save_figure(save_str)

def visualize_gp_and_sample_path(gp, domain, data, sample_path, n_grid=500,
                                 n_gen=20000, n_samp_plot=10, std_mult=3,
                                 show_sp=False, ylim=[-4, 4], save_str=None):
    """Call both visualize_gp and visualize_sample_path_and_data."""
    update_rc_params()

    visualize_gp(gp, domain, data, n_grid, n_gen, n_samp_plot, std_mult,
                 show_sp, ylim)
    visualize_sample_path_and_data(sample_path, data, ylim)

    # Optionally save figure 
    save_figure(save_str)

def update_rc_params():
    """Update matplotlib rc params."""
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})

def set_plot_and_show(xlim, ylim, xlabel='x', ylabel='y', plot_width=9,
                      plot_height=5):
    """Set various plot properties and call plt.show()."""
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gcf().set_size_inches(plot_width, plot_height)

    plt.show()

def save_figure(save_str):
    if save_str is not None:
        file_name = save_str + '.png'
        plt.savefig(file_name, bbox_inches='tight')

        # Print save message
        print('Saved figure: {}'.format(file_name))
