"""
Code for Gaussian processes.
"""


class SimpleGp:
    """
    Simple GP model without external backend.
    """

    def __init__(self, params=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        verbose : bool
            If True, print description string.
        """
        self.set_params(params)
        if verbose:
            self.print_str()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""

        # If params is dict, convert to Namespace
        if isinstance(params, dict):
            params = Namespace(**params)

        # Set self.params
        self.params = Namespace()
        self.params.ls = getattr(params, 'ls', 3.7)
        self.params.alpha = getattr(params, 'alpha', 1.85)
        self.params.sigma = getattr(params, 'sigma', 1e-5)
        self.params.kernel = getattr(params, 'kernel', kern_exp_quad)

    def set_data(self, data):
        """Set self.data."""
        self.data_init = copy.deepcopy(data)
        self.data = copy.deepcopy(self.data_init)

    def inf(self, data):
        """Set data, run inference, update self.sample_list."""
        self.set_data(data)
        self.sample_list = [
            Namespace(
                ls=self.params.ls, alpha=self.params.alpha, sigma=self.params.sigma
            )
        ]

    def post(self, s):
        """Return one posterior sample."""
        return self.sample_list[0]

    def gen_list(self, x_list, z, s, nsamp):
        """
        Draw nsamp samples from generative process, given list of inputs
        x_list, posterior sample z, and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        z : Namespace
            Namespace of GP hyperparameters.
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from generative process.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        pred_list = self.sample_gp_pred(nsamp, x_list, z)
        return pred_list

    def postgen_list(self, x_list, s, nsamp):
        """
        Draw nsamp samples from posterior predictive distribution, given list
        of inputs x_list and seed s.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).
        s : int
            The seed, a positive integer.
        nsamp : int
            The number of samples to draw from the posterior predictive
            distribution.

        Returns
        -------
        list
            A list with len=len(x_list) of numpy ndarrays, each with
            shape=(nsamp,).
        """
        hp = self.sample_list[0]
        pred_list = self.sample_gp_post_pred(nsamp, x_list, hp, full_cov=True)
        return pred_list

    def sample_gp_pred(self, nsamp, input_list, hp):
        """
        Sample from GP predictive distribution given one posterior GP sample.

        Parameters
        ----------
        nsamp : int
            Number of samples from predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(-1, ).
        hp : Namespace
            Namespace of GP hyperparameters.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        postmu, postcov = gp_post(
            self.data.X,
            self.data.y,
            input_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
        )
        single_post_sample = sample_mvn(postmu, postcov, 1).reshape(-1)

        pred_list = [
            single_post_sample for _ in range(nsamp)
        ]
        return list(np.stack(pred_list).T)

    def sample_gp_post_pred(self, nsamp, input_list, hp, full_cov=False):
        """
        Sample from GP posterior predictive distribution.

        Parameters
        ----------
        nsamp : int
            Number of samples from posterior predictive distribution.
        input_list : list
            A list of numpy ndarray shape=(-1, ).
        hp : Namespace
            Namespace of GP hyperparameters.
        full_cov : bool
            If True, return covariance matrix, else return diagonal only.

        Returns
        -------
        list
            A list of len=len(input_list) of numpy ndarrays shape=(nsamp, 1).
        """
        postmu, postcov = self.get_gp_post_params_list(input_list, hp, full_cov)

        if full_cov:
            ppred_list = list(sample_mvn(postmu, postcov, nsamp))
        else:
            ppred_list = list(
                np.random.normal(
                    postmu.reshape(-1,),
                    postcov.reshape(-1,),
                    size=(nsamp, len(input_list)),
                )
            )

        return list(np.stack(ppred_list).T)

    def get_gp_post_params_list(self, x_list, hp=None, full_cov=True):
        """
        Compute and return GP posterior parameters for list of inputs x_list.
        """
        if hp is None:
            hp = self.sample_list[0]

        postmu, postcov = gp_post(
            self.data.X,
            self.data.y,
            input_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
            full_cov,
        )

        return postmu, postcov

    def get_gp_post_list(self, x_list):
        """
        Get GP posterior for each input x in x_list, and return list of means
        and list of stds (on element for each x in x_list).

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays each with shape=(-1,).

        Returns
        -------
        list
            A list with len=len(x_list) of floats, corresponding to posterior
            mean for each x in x_list.
        list
            A list with len=len(x_list) of floats, corresponding to posterior
            stds for each x in x_list.
        """
        hp = self.sample_list[0]
        postmu, postcov = gp_post(
            self.data.X,
            self.data.y,
            x_list,
            hp.ls,
            hp.alpha,
            hp.sigma,
            self.params.kernel,
            full_cov=False,
        )
        postmu_list = [x for x in postmu]
        poststd_list = [x for x in postcov]
        return postmu_list, poststd_list

    def get_gp_post_single(self, x):
        """
        Get GP posterior for an input x. Return posterior mean and std for x.
        """
        postmu_list, poststd_list = self.get_gp_post_list([x])
        return postmu_list[0], poststd_list[0]

    def print_str(self):
        """Print a description string"""
        print('*SimpleGp with params={}'.format(self.params))
