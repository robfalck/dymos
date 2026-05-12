import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander


def birkhoff_matrix(tau, w, grid_type):
    """
    Return the pseudospectral integration matrix for a Birkhoff polynomial at the given nodes.

    Parameters
    ----------
    tau : ndarray[:]
        Vector of given nodes in the polynomial.
    w : ndarray[:]
        Vector of nodes at which the polynomial is evaluated.
    grid_type : str
        The type of Gaussian grid used in the transcription.

    Returns
    -------
    np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the interpolation of state values
        at the interior LGL/CGL nodes.
    """
    N = tau.size - 1
    end_node = N if tau[-1] == 1.0 else N + 1

    alpha = np.zeros((N + 1, N + 1))
    S = np.zeros((N + 1, N + 1))

    # compute the pseudo-Vandermonde matrices for the required max degrees
    if grid_type[0] == 'l':
        # need up to degree N for alpha, and up to N+1 for S
        V_alpha = legvander(tau, N)
        V_S = legvander(tau[1:], N + 1)
    elif grid_type[0] == 'c':
        V_alpha = chebvander(tau, N)
        V_S = chebvander(tau[1:], N + 1)
    else:
        raise ValueError('invalid grid type')

    # vectorize alpha assembly via Vandermonde transpose
    alpha[:end_node, :] = w * V_alpha[:, :end_node].T

    if grid_type == 'lgl':
        alpha[N, :] = N * V_alpha[:, N] * w / (2 * N + 1)
    elif grid_type == 'cgl':
        alpha[N, :] = V_alpha[:, N] * w / 2

    # vectorize S assembly via matrix slicing
    if grid_type[0] == 'l':
        S[1:, 0] = (tau[1:] - tau[0]) / 2

        # slicing V_S[:, 2:] gives P_{n+1}, V_S[:, :-2] gives P_{n-1} for n=1 to N
        S[1:, 1:] = (V_S[:, 2:] - V_S[:, :-2]) / 2

    elif grid_type[0] == 'c':
        S[1:, 0] = (tau[1:] - tau[0]) / np.pi
        S[1:, 1] = (tau[1:]**2 - tau[0]**2) / np.pi

        n_S = np.arange(2, N + 1)

        # P_{n+1} is V_S[:, 3:], P_{n-1} is V_S[:, 1:-2] for n=2 to N
        int_p = (V_S[:, 3:] / (2 * n_S + 2) -
                 V_S[:, 1:-2] / (2 * n_S - 2) -
                 (-1.0)**n_S / (n_S**2 - 1))

        S[1:, 2:] = int_p / (np.pi / 2)

    B = S @ alpha

    return B
