import itertools

import numpy as np


class LagrangeBarycentricInterpolant(object):
    """
    Class definition for LagrangeBarycentricInterpolant.

    Interpolate values and first derivatives of set of data using barycentric interpolation of a
    Lagrange Polynomial.

    Parameters
    ----------
    nodes : sequence
        The nodes of the polynomial from -1 to 1 on which the values to
        be interpolated are given.
    shape : tuple
        Shape of the values to be interpolated.

    Attributes
    ----------
    num_nodes : int
        The number of nodes in the interpolated polynomial.
    tau_i : np.array
        The location of the nodes of the polynomial in tau space [-1 1].
    x_0 : float
        The x-value corresponding to the left-side of the interval (tau = -1).
    x_f : float
        The x-value corresponding to the right-side of the interval (tau = +1).
    f_j : np.array
        The values to be interpolated
    w_b : np.array
        The barycentric weights for the points in the interpolated polynomial.
    wbfj : np.array
        An array of the precomputed product of the interpolated values and
        the corresponding barycentric weights.
    dx_dtau : float
        Half the span from x0 to xf.  The ratio of x-space to
        internal tau-space.

    Notes
    -----
    The Barycentric formula is given in Eq. 3.3 of [1]_ as

    .. math::

        p(x) = l(x) \\Sum \\frac{w_j f_j}{x-x_j}

    where l(x) is

    .. math::

        l(x) = (x-x_0)(x-x_1)(x-x_2)...

    The singularity in the denominator of p(x) at x = x_n is cancelled
    by the the term (x-x_n) in l(x).

    References
    ----------
    .. [1] Berrut, Jean-Paul, and Lloyd N. Trefethen.
       "Barycentric lagrange interpolation." Siam Review 46.3 (2004): 501-517.
    """

    def __init__(self, nodes, shape):

        self.num_nodes = len(nodes)
        """ The number of nodes in the interpolated polynomial. """

        self.tau_i = nodes
        """ The independent variable values at interpolation points. """

        _shape = (self.num_nodes,) + shape

        self.f_j = np.zeros(_shape)
        """ An array of values to be interpolated. """

        self.w_b = np.ones(self.num_nodes)
        """ Barycentric weights for nodes in the interpolated polynomial."""

        # Barycentric Weights
        for j in range(self.num_nodes):
            for k in range(self.num_nodes):
                if k != j:
                    self.w_b[j] /= (self.tau_i[j] - self.tau_i[k])

        self.wbfj = np.zeros(_shape)
        """ An array of the precomputed product of the interpolated
            values and the corresponding barycentric weights.
        """

        n = self.wbfj.shape[0]
        m = np.prod(self.wbfj.shape[1:])
        self.wbfj_flat = np.reshape(self.wbfj, newshape=(n, m))
        """ A flattened view of wbfj"""

        self.x0 = -1.0
        """ The value of the independent axis corresponding to $\tau = -1$ """

        self.xf = 1.0
        """ The value of the independent axis corresponding to $\tau = 1$ """

        self.dx_dtau = 1.0
        """ Half the span from x0 to xf.  The ratio of x-space to
        internal tau-space. """

        self._is_setup = False

    def x_to_tau(self, x):
        """
        Converts the independent variable x to its corresponding value of $\tau$.

        Given bounds on the independent variable x0 and xf which
        correspond to $\tau$ of -1 and 1, respectively, the returned value
        will be the equivalent $tau$.

        For instance, if x0 = 0 and xf = 100, x = 50 will have an equivalent
        $\tau$ of 0 (halfway on [-1 1]).

        Parameters
        ----------
        x : float or ndarray
            The independent variable to be converted to $\tau$.

        Returns
        -------
        float or ndarray
            The equivalent value of $\tau$ given x.
        """
        return -1.0 + (x - self.x0) / self.dx_dtau

    def setup(self, x0, xf, f_j):
        """
        Prepare the interpolant for use by setting the values to be interpolated.

        Parameters
        ----------
        x0 : float
            The lower bound of the independent variable.
            corresponding to $\tau = -1$ .
        xf : float
            The upper bound of the independent variable.
            corresponding to $\tau = -1$ .
        f_j : np.array
            The values to be interpolated at the nodes on [-1 1].

        Raises
        ------
        ValueError
            If the length of f_j is not the number of nodes in the interpolated
            polynomial.
        """
        if len(f_j) != self.num_nodes:
            raise ValueError(f"f_j must have {self.num_nodes} values")
        self.f_j[...] = f_j
        self.x0 = x0
        self.xf = xf
        self.dx_dtau = 0.5 * (xf - x0)

        fjT = self.f_j.T
        self.wbfj[...] = (self.w_b * fjT).T
        self._is_setup = True

    def eval(self, x):
        """
        Interpolate the polynomial at x.

        Parameters
        ----------
        x : float
            The independent variable value at which interpolation
            is requested.

        Returns
        -------
        float
            The interpolated value of the polynomial at x.
        """
        if not self._is_setup:
            raise RuntimeError('LagrangeBarycentricInterpolant has not been setup')
        tau = self.x_to_tau(x)

        g = tau - self.tau_i
        l = np.ones_like(g)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if j != i:
                    l[i] *= g[j]

        result = np.reshape(np.dot(l, self.wbfj_flat), newshape=self.wbfj.shape[1:])

        return result

    def eval_deriv(self, x, der=1):
        """
        Interpolate the derivative of the polynomial at x.

        Parameters
        ----------
        x : float
            The independent variable value at which the derivative
            is requested.
        der : int
            Derivative order requested. Default is 1 for first derivatives.

        Returns
        -------
        float
            The first derivative of the polynomial at x.
        """
        if not self._is_setup:
            raise RuntimeError('LagrangeBarycentricInterpolant has not been setup')
        if der >= self.num_nodes:
            return 0.0

        n = self.num_nodes
        tau = self.x_to_tau(x)
        g = tau - self.tau_i
        lprime = np.zeros(n)

        if der == 1:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    prod = 1.0
                    for k in range(n):
                        if k != i and k != j:
                            prod *= g[k]
                    lprime[i] += prod
            # df_dtau = np.dot(lprime, self.wbfj)
            df_dtau = np.reshape(np.dot(lprime, self.wbfj_flat), newshape=self.wbfj.shape[1:])
            return df_dtau / self.dx_dtau
        elif der == 2:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        prod = 1.0
                        for ii in range(n):
                            if ii != i and ii != j and ii != k:
                                prod *= g[ii]
                        lprime[i] += prod
            df_dtau = np.reshape(np.dot(lprime, self.wbfj_flat), newshape=self.wbfj.shape[1:])
            return df_dtau / self.dx_dtau**2
        else:
            raise ValueError('Barycentric interpolant currently only supports up to '
                             'second derivatives')

import openmdao.api as om
class BarycentricInterpComp(om.ExplicitComponent):
    """
    Class definition for LagrangeBarycentricInterpolant2.

    Interpolate values and first derivatives of set of data using barycentric interpolation of a
    Lagrange Polynomial.

    Parameters
    ----------
    nodes : sequence
        The nodes of the polynomial from -1 to 1 on which the values to
        be interpolated are given.
    shape : tuple
        Shape of the values to be interpolated.

    Attributes
    ----------
    num_nodes : int
        The number of nodes in the interpolated polynomial.
    tau_i : np.array
        The location of the nodes of the polynomial in tau space [-1 1].
    x_0 : float
        The x-value corresponding to the left-side of the interval (tau = -1).
    x_f : float
        The x-value corresponding to the right-side of the interval (tau = +1).
    f_j : np.array
        The values to be interpolated
    w_b : np.array
        The barycentric weights for the points in the interpolated polynomial.
    wbfj : np.array
        An array of the precomputed product of the interpolated values and
        the corresponding barycentric weights.
    dx_dtau : float
        Half the span from x0 to xf.  The ratio of x-space to
        internal tau-space.

    Notes
    -----
    The Barycentric formula is given in Eq. 3.3 of [1]_ as

    .. math::

        p(x) = l(x) \\Sum \\frac{w_j f_j}{x-x_j}

    where l(x) is

    .. math::

        l(x) = (x-x_0)(x-x_1)(x-x_2)...

    The singularity in the denominator of p(x) at x = x_n is cancelled
    by the term (x-x_n) in l(x).

    References
    ----------
    .. [1] Berrut, Jean-Paul, and Lloyd N. Trefethen.
       "Barycentric lagrange interpolation." Siam Review 46.3 (2004): 501-517.
    """
    def __init__(self, nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tau_i = nodes

        self._num_nodes = n = len(self._tau_i)

        self._l_idxs = list(itertools.combinations(range(n)[::-1], n - 1))

        # Instead of using multiply-nested for loops in compute_partials
        # to assign jacobian elements, pre-compute indices into the
        # jacobians so that we can populate them within a single for loop.
        self._del_dg_idxs = []
        self._d2el_dg2_idxs = []

        for i in range(n):
            for j in range(i):
                self._del_dg_idxs.append((i, j))

        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(j):
                    if k == i:
                        continue
                    self._d2el_dg2_idxs.append((i, j, k))

        self._interpolants = {}

    def initialize(self):
        self.options.declare('time_units', types=(str,), default='s',
                             desc='Units of time for the interpolated rates.')

    def setup(self):
        self._w_b = np.ones(self._num_nodes)
        """ Barycentric weights for nodes in the interpolated polynomial."""

        # Barycentric Weights
        for j in range(self._num_nodes):
            for k in range(self._num_nodes):
                if k != j:
                    self._w_b[j] /= (self._tau_i[j] - self._tau_i[k])

        self.add_input('t', units=self.options['time_units'],
                       desc='Value of time at which interpolation should be performed.')
        self.add_input('t0', units=self.options['time_units'],
                       desc='Initial value of time in the interpolation interval.')
        self.add_input('dt_dtau', units=self.options['time_units'],
                       desc='Ratio of integration timespan to non-dimensional (tau) span for the '
                            'interpolation interval.')

    def add_interpolant(self, name, shape, units):
        self.add_input(f'inputs:{name}', shape=(self._num_nodes,) + shape, units=units,
                       desc=f'Values of {name} at the given interpolation nodes.')

        self.add_output(f'interp:{name}', shape=shape, units=units,
                        desc=f'Value of {name} at the given time.')

        self._interpolants[name] = {'shape': shape,
                                    'units': units,
                                    'input_name': f'inputs:{name}',
                                    'output_name': f'interp:{name}'}

        self.declare_partials(of=self._interpolants[name]['output_name'],
                              wrt=self._interpolants[name]['input_name'])

        self.declare_partials(of=self._interpolants[name]['output_name'],
                              wrt='t')

        self.declare_partials(of=self._interpolants[name]['output_name'],
                              wrt='t0')

        self.declare_partials(of=self._interpolants[name]['output_name'],
                              wrt='dt_dtau')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        t = inputs['t']

        # First, a memory intensive way
        # Build a matrix G that contains the difference between
        # the given time and time and all of the nodes (columns), then
        # repeated over every row.
        # We will then set the diagonal of this matrix to be equal to one.
        tau = -1 + (t - inputs['t0']) / inputs['dt_dtau']
        g = tau - self._tau_i
        # G = np.repeat(g, 5).reshape((5, 5))
        # np.fill_diagonal(G, 1.0)

        el = np.prod(g[self._l_idxs], axis=1)
        # print(el)
        # exit(0)
        # el = np.prod(G, axis=0)

        for interp_options in self._interpolants.values():
            iname = interp_options['input_name']
            oname = interp_options['output_name']
            f_j = inputs[iname]

            # Multiply w_b and f_j along the first axis of f_j
            wbfj = np.swapaxes(np.swapaxes(f_j, 0, -1) * self._w_b, -1, 0)

            # Multiply el and wbfj along the first axis of wbfj
            outputs[oname] = np.einsum('i,ij...->j...', el, wbfj)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self._num_nodes
        t = inputs['t']

        # First, a memory intensive way
        # Build a matrix G that contains the difference between
        # the given time and time and all of the nodes (columns), then
        # repeated over every row.
        # We will then set the diagonal of this matrix to be equal to one.
        tau = -1 + (t - inputs['t0']) / inputs['dt_dtau']
        g = tau - self._tau_i
        el = np.prod(g[self._l_idxs], axis=1)

        dtau_dt = 1 / inputs['dt_dtau']
        dtau_dt0 = -dtau_dt
        dtau_ddt_dtau = -(t - inputs['t0']) * dtau_dt ** 2

        del_dg = np.zeros((n, n))
        d2el_dg2 = np.zeros((n, n, n))

        for i, j in self._del_dg_idxs:
            del_dg[i, j] = del_dg[j, i] = np.prod(np.delete(g, [i, j]))

        for i, j, k in self._d2el_dg2_idxs:
            d2el_dg2[i, j, k] = d2el_dg2[i, k, j] = np.prod(np.delete(g, [i, j, k]))


        print('number of iterations for d2el_dg2', len(self._d2el_dg2_idxs))
        print('total size of d2el_dg2', d2el_dg2.size)
        print('nonzeros in d2el_dg2', np.count_nonzero(d2el_dg2))
        print('density of d2el_dg2', np.count_nonzero(d2el_dg2)/d2el_dg2.size)
        print('unique values in d2el_dg2', len(set(d2el_dg2.ravel())))
        print(tuple([tuple(sorted(idxs)) for idxs in self._d2el_dg2_idxs]))
        print(len(set(tuple([tuple(sorted(idxs)) for idxs in self._d2el_dg2_idxs]))))

        # for i in range(n):
        #     for j in range(n):
        #         if j == i:
        #             continue
        #         for k in range(j):
        #             if k == i:
        #                 continue
        #             del_dg2[i, j, k] = np.prod(np.delete(g, [i, j, k]))
        #             del_dg2[i, k, j] = del_dg2[i, j, k]

        # # The nonzero (off-diagonal) elements of del_dg.
        # del_dg_nz = np.prod(g[self._del_dg_idxs], axis=2)
        #
        # for i in range(n):
        #     # Get a view of the ith row of del_dg with the diagonal omitted
        #     del_dg_row_i = np.delete(del_dg[i, ...], i, axis=0)
        #     # Assign our nonzero data to that row (except for the diagonal)
        #     del_dg_row_i[:] = del_dg_nz[i, :]
        #     print(del_dg_row_i)
        #
        # print(del_dg)

        # with np.printoptions(linewidth=1024, edgeitems=1024):
        #     print(d2el_dg2)
        exit(0)

        d_wbfj_dfj = self._w_b

        for interp_options in self._interpolants.values():
            iname = interp_options['input_name']
            oname = interp_options['output_name']
            f_j = inputs[iname]

            # Multiply w_b and f_j along the first axis of f_j
            wbfj = np.swapaxes(np.swapaxes(f_j, 0, -1) * self._w_b, -1, 0)

            # Multiply el and wbfj along the first axis of wbfj
            # outputs[interp_options['output_name']] = np.einsum('i,ij->j', el, wbfj)

            d_interp_result_d_el = wbfj.T
            d_interp_result_d_wbfj = el

            # print('d_interp_result_d_el')
            # print(d_interp_result_d_el)
            #
            # print('d_interp_result_d_wbfj')
            # print(d_interp_result_d_wbfj)

            # partials[oname, 't'] = d_interp_result_d_el * del_dg * dg_dtau * dtau_dt
            #                            (1, 5)            (5, 5)   (5, 1)    (1, 1)

            # Since dg_dtau is an array of 1's where we'd typically do a dot product
            # below with something like
            # np.dot(np.dot(d_interp_result_d_el, del_dg), dg_dtau)
            # we instead do
            # np.sum(np.dot(d_interp_result_d_el, del_dg))
            # which can be converted to the einsum
            # np.einsum('ij,ij->', d_interp_result_d_el, del_dg)
            d_interp_result_dg = np.einsum('ij,ij->', d_interp_result_d_el, del_dg)
            partials[oname, 't'] = d_interp_result_dg * dtau_dt
            partials[oname, 't0'] = d_interp_result_dg * dtau_dt0
            partials[oname, 'dt_dtau'] = d_interp_result_dg * dtau_ddt_dtau
            partials[oname, iname] = d_interp_result_d_wbfj * d_wbfj_dfj


if __name__ == '__main__':
    from dymos.utils.lgl import lgl

    tau, w = lgl(7)

    p = om.Problem()
    interp_comp = BarycentricInterpComp(nodes=tau)
    p.model.add_subsystem('interp_comp', interp_comp, promotes=['*'])

    t = tau + 1
    y = t ** 2

    interp_comp.add_interpolant('y', shape=(1,), units='m')

    p.setup(force_alloc_complex=True)

    p.set_val('t0', 0.0)
    p.set_val('dt_dtau', 1.0)
    p.set_val('inputs:y', y)

    for t in np.linspace(0, 2, 10):
        p.set_val('t', t)
        p.run_model()
        y = p.get_val('interp:y')
        print(t**2, y)

    p.set_val('t', 0.1)
    p.run_model()

    with np.printoptions(linewidth=1024):
        p.check_partials(method='cs')



