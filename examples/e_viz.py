from argparse import Namespace
import copy
import numpy as np



np.random.seed(11)


def make_viz():
    """Produce visualization."""
    # Set plot settings
    ylim = [-5, 5]
    gpv.update_rc_params()

    # Define model params
    gp1_hypers = {'ls': 1., 'alpha': 1.5, 'sigma': 1e-1}
    gp2_hypers = {'ls': 3., 'alpha': 1., 'sigma': 1e-1}
    #gp2_hypers = {'ls': 1., 'alpha': 1.5, 'sigma': 1e-1}

    # Define domain
    x_min = -10.
    x_max = 10.
    domain = RealDomain({'min_max': [(x_min, x_max)]})

    # Drawn one sample path from gp1_prior
    gp1_prior = MySimpleGp(params=gp1_hypers)
    data_priorish = Namespace(X=[np.array([-50.])], y=np.array([0.]))
    gp1_prior.inf(data_priorish)
    sample_path = gput.get_sample_path(gp1_prior, domain)


    n_obs_list = [3, 5, 10, 15, 17, 25, 40]

    # Make full data
    data_full = get_data_from_sample_path(sample_path, 100)

    # TODO: set domain_list
    domain_list = list(np.linspace(x_min, x_max, 200))

    # TODO: loop through n_obs_list
    for n_obs in n_obs_list:

        # TODO: define dataset for n_obs
        data = get_data_subset(data_full, n_obs)

        # TODO: define gp_num (posterior GP, do inference on a set of data)
        gp_num = #

        # TODO: define gp_den (modified prior GP, based on output variance [alpha])
        gp_den = #

        lb_list, ub_list = get_lb_ub_lists(domain_list, gp_num, gp_den)

        # TODO: various plotting


def get_lb_ub_lists(domain_list, gp_num, gp_den): 
    """Return lb_list and ub_list given domain_list."""

    lb_list, ub_list = [], []

    for x in domain_list:
        print('domain point x: {}'.format(x))

        x_list = data.X + [np.array([x])]
        
        gp_num_params = gp_num.get_params(x_list)
        gp_den_params = gp_den.get_params(x_list)

        lb, ub = get_conf_bounds(gp_num_params, gp_den_params, ellipse=True,
                                 normal=True, mult_std=None)

        lb_list.append(lb)
        ub_list.append(ub)

    return lb_list, ub_list


def get_conf_bounds(gp_num_params, gp_den_params, normal=True, mult_std=3):
    """Return lower bound and upper bound."""
    alpha_level = .05

    quot_mean, quot_cov = normal_quotient(gp_num_params['mean'], gp_num_params['cov'],
                                          gp_den_params['mean'], gp_den_params['cov'])

    log_z = normal_quotient_log_z(gp_num_params['mean'], gp_num_params['cov'],
                                  gp_den_params['mean'], gp_den_params['cov'])

    log_quot_at_mean = get_log_mvn_pdf_at_mean(quot_mean, quot_cov)
    log_alpha = np.log(alpha_level)

    m_dist = np.sqrt(2 * (log_quot_at_mean - log_z - log_alpha))

    # Constrain Mahalanobis distance
    m_dist_min = .1
    m_dist_max = 10.
    m_dist = m_dist_min if m_dist < m_dist_min else m_dist
    m_dist = m_dist_max if m_dist > m_dist_max else m_dist

    max_projection = np.sqrt(quot_cov[-1, -1]) * m_dist

    lb = quot_mean[-1] - max_projection
    ub = quot_mean[-1] + max_projection

    return lb, ub









    #alpha = 1.01
    ##alpha = 1.001
    ##alpha = 1.00001
    
    ## Define modified prior
    #gp3_hypers = copy.deepcopy(gp2_hypers)
    #gp3_hypers['alpha'] = alpha

    ## Compute gp2_post
    #data = get_data_subset(data_full, n_obs)
    #gp2_post = MySimpleGp(params=gp2_hypers)
    #gp2_post.inf(data)

    ## Visualize sample_path, data, and GP posterior
    #plt.figure()

    #domain_pt_arr = np.linspace(x_min, x_max, 200)

    #quot_low_list, quot_up_list, quot_mean_list = [], [], []
    #quot_low_2_list, quot_up_2_list = [], []
    #for domain_pt in domain_pt_arr:
        #print('domain pt: {}'.format(domain_pt))

        #x_list = data.X + [np.array([domain_pt])]

        #gp3_prior_mean, gp3_prior_cov = get_gp_prior_params(x_list,
                                                            #gp3_hypers['ls'],
                                                            #gp3_hypers['alpha'],
                                                            #gp2_post.params.kernel) # kernel type is the same
        #gp3_prior_params = {'mean': gp3_prior_mean, 'cov': gp3_prior_cov}


        #gp2_post_mean, gp2_post_cov = gp2_post.get_gp_post_params_list(x_list)
        #gp2_post_params = {'mean': gp2_post_mean, 'cov': gp2_post_cov}

        #quot_low, quot_up, quot_low_2, quot_up_2 = get_quot_bounds(gp2_post_params, gp3_prior_params,
                                            #ellipse=True, normal=True,
                                            #mult_std=None)

        #quot_low_list.append(quot_low)
        #quot_up_list.append(quot_up)
        #quot_mean_list.append(.5 * (quot_low + quot_up))

        #quot_low_2_list.append(quot_low_2)
        #quot_up_2_list.append(quot_up_2)

    ## Plot C_t bounds filling
    #plt.fill_between(domain_pt_arr, quot_low_list, quot_up_list,
                     #color='wheat', alpha=0.5)

    ## Plot C_t bounds
    #plt.plot(domain_pt_arr, quot_low_list, 'r--')
    #plt.plot(domain_pt_arr, quot_up_list, 'r--')

    ## Plot C_t bounds
    #plt.plot(domain_pt_arr, quot_low_2_list, 'g--')
    #plt.plot(domain_pt_arr, quot_up_2_list, 'g--')

    ## Plot GP posterior
    #save_str = 'viz_' + str(n_obs)
    #gpv.visualize_gp_and_sample_path(gp2_post, domain, data, sample_path,
                                     #show_sp=False, ylim=ylim,
                                     #save_str=save_str)
