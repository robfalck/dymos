import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.chebyshev import chebvander
import scipy.special as sp
import matplotlib.pyplot as plt

_lgl_cache = {}
""" Cache for the LGL nodes and weights, keyed by n. """


def _lgl(n, tol=np.finfo(float).eps):
    """
    Returns the Legendre-Gauss-Lobatto nodes and weights for a Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Based on the routine written by Greg von Winckel (License follows)

    Copyright (c) 2009, Greg von Winckel
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR aux_outputs
    PARTICULAR PURPOSE  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    n : int
        The number of LGL nodes requested.  The order of the polynomial is n-1.
    tol : float
        The tolerance to which the location of the nodes should be converged.

    Returns
    -------
    x : numpy.array
        An array of the LGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LGL weights at the nodes in x.
    """
    n = n - 1
    n1 = n + 1
    n2 = n + 2
    # Get the initial guesses from the Chebyshev nodes
    x = np.cos(np.pi * (2 * np.arange(1, n2) - 1) / (2 * n1))
    P = np.zeros([n1, n1])
    # Compute P_(n) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2

    for i in range(100):
        if np.all(np.abs(x - xold) <= tol):
            break
        xold = x
        P[:, 0] = 1.0
        P[:, 1] = x

        for k in range(2, n1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        x = xold - (x * P[:, n] - P[:, n - 1]) / (n1 * P[:, n])
    else:
        raise RuntimeError(f"Failed to converge LGL nodes for order {n}")

    x.sort()

    w = 2 / (n * n1 * P[:, n] ** 2)

    return x, w


def lgl(n):
    """
    Retrieve the lgl nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with lgl nodes and weights.
    """
    if n not in _lgl_cache:
        _lgl_cache[n] = _lgl(n)
    return _lgl_cache[n]


_cgl_cache = {}
""" Cache for the LGL nodes and weights, keyed by n. """


def clenshaw_curtis(n):
    """
    Returns the Chebyshev-Gauss-Lobatto nodes and weights for a Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Parameters
    ----------
    n : int
        The number of LGL nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the LGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CGL weights at the nodes in x.
    """
    x = -np.cos(np.pi * np.arange(n) / (n - 1))
    w = np.zeros(n)
    even = False if n % 2 else True

    N_2 = int(n / 2) if even else int((n + 1) / 2)
    j_lim = N_2 if even else int((n - 1) / 2)

    for k in range(n):
        c = 1 if k == 0 else 2
        s = 0
        for j in range(1, N_2):
            b = 1 if j == j_lim else 2
            s += b * np.cos(2 * j * k * np.pi / (n - 1)) / (4 * j**2 - 1)
        w[k] = c * (1 - s) / (n - 1)

    if even:
        w[N_2:] = np.flip(w[:N_2])
    else:
        w[N_2:] = np.flip(w[: (N_2 - 1)])

    return x, w


def _cgl(n):
    """
    Returns the Chebyshev-Gauss-Lobatto nodes and weights for a Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Parameters
    ----------
    n : int
        The number of CGL nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the CGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CGL weights at the nodes in x.
    """
    x = -np.cos(np.pi * np.arange(n) / (n - 1))
    w = np.pi / (n - 1) * np.ones(n)
    w[0] *= 0.5
    w[-1] *= 0.5

    return x, w


def cgl(n):
    """
    Retrieve the cgl nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with cgl nodes and weights.
    """
    if n not in _cgl_cache:
        _cgl_cache[n] = _cgl(n)
    return _cgl_cache[n]


def birkhoff_matrix_vander(tau, w, grid_type):
    """
    Returns the pseudospectral integration matrix for a Birkhoff polynomial at the given nodes.
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
    if grid_type[0] == "l":
        # need up to degree N for alpha, and up to N+1 for S
        V_alpha = legvander(tau, N)
        V_S = legvander(tau[1:], N + 1)
    elif grid_type[0] == "c":
        V_alpha = chebvander(tau, N)
        V_S = chebvander(tau[1:], N + 1)
    else:
        raise ValueError("invalid grid type")

    # vectorize alpha assembly via Vandermonde transpose
    alpha[:end_node, :] = w * V_alpha[:, :end_node].T

    if grid_type == "lgl":
        alpha[N, :] = N * V_alpha[:, N] * w / (2 * N + 1)
    elif grid_type == "cgl":
        alpha[N, :] = V_alpha[:, N] * w / 2

    # vectorize S assembly via matrix slicing
    if grid_type[0] == "l":
        S[1:, 0] = (tau[1:] - tau[0]) / 2

        # slicing V_S[:, 2:] gives P_{n+1}, V_S[:, :-2] gives P_{n-1} for n=1 to N
        S[1:, 1:] = (V_S[:, 2:] - V_S[:, :-2]) / 2

    elif grid_type[0] == "c":
        S[1:, 0] = (tau[1:] - tau[0]) / np.pi
        S[1:, 1] = (tau[1:] ** 2 - tau[0] ** 2) / np.pi

        n_S = np.arange(2, N + 1)

        # P_{n+1} is V_S[:, 3:], P_{n-1} is V_S[:, 1:-2] for n=2 to N
        int_p = (
            V_S[:, 3:] / (2 * n_S + 2) - V_S[:, 1:-2] / (2 * n_S - 2) - (-1.0) ** n_S / (n_S**2 - 1)
        )

        S[1:, 2:] = int_p / (np.pi / 2)

    B = S @ alpha

    return B


def birkhoff_matrix(tau, w, grid_type):
    """
    Returns the pseudospectral integration matrix for a Birkhoff polynomial at the given nodes.
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
        at the interior LGL nodes.
    """
    N = tau.size - 1
    end_node = N if tau[-1] == 1 else N + 1
    alpha = np.zeros((N + 1, N + 1))
    S = np.zeros((N + 1, N + 1))
    if grid_type[0] == "l":
        pol = sp.eval_legendre
    elif grid_type[0] == "c":
        pol = sp.eval_chebyt
    else:
        raise ValueError("invalid grid type")
    for j in range(0, N + 1):
        for n in range(0, end_node):
            alpha[n, j] = w[j] * pol(n, tau[j])
        if grid_type == "lgl":
            alpha[N, j] = N * pol(N, tau[j]) * w[j] / (2 * N + 1)
        elif grid_type == "cgl":
            alpha[N, j] = pol(N, tau[j]) * w[j] / 2
    if grid_type[0] == "l":
        for i in range(1, N + 1):  # The first row is exactly zero.
            S[i, 0] = (tau[i] - tau[0]) / 2
            for n in range(1, N + 1):
                gamma = 2 / (2 * n + 1)
                int_p = (pol(n + 1, tau[i]) - pol(n - 1, tau[i])) / (2 * n + 1)
                S[i, n] = int_p / gamma
    elif grid_type[0] == "c":
        gamma = np.pi / 2
        for i in range(1, N + 1):  # The first row is exactly zero.
            # chebyshev polynomial of order 0: 1
            S[i, 0] = (tau[i] - tau[0]) / np.pi
            # chebyshev polynomial of order 1: x
            S[i, 1] = (tau[i] ** 2 - tau[0] ** 2) / np.pi
            for n in range(2, N + 1):
                int_p = (
                    pol(n + 1, tau[i]) / (2 * n + 2)
                    - pol(n - 1, tau[i]) / (2 * n - 2)
                    - (-1) ** n / (n**2 - 1)
                )
                S[i, n] = int_p / gamma
    B = S @ alpha
    return B


time_old = []
time_new = []
error = []
tau_gen = {}
tau_gen["lgl"] = lgl
tau_gen["cgl"] = cgl
res = {}

start = 10
stop = 2000
step = 100

for tp in ["lgl", "cgl"]:
    time_old = []
    time_new = []
    error = []
    for _ in range(start, stop, step):
        x, w = cgl(_)
        from time import perf_counter as pc

        t1 = pc()
        B1 = birkhoff_matrix(x, w, grid_type=tp)
        t2 = pc()
        t3 = pc()
        B2 = birkhoff_matrix_vander(x, w, grid_type=tp)
        t4 = pc()
        error.append(np.linalg.norm(B1 - B2, 2))
        time_old.append((t2 - t1))
        time_new.append((t4 - t3))
    res[tp] = (np.array(time_old), np.array(time_new), np.array(error))

n = np.array(list(range(start, stop, step)))

for k, v in res.items():
    fg, ax = plt.subplots(1, 3, figsize=(15, 4))
    fg.suptitle(k.upper())
    ax[0].loglog(n, v[2])
    ax[0].set_ylabel("error (2-norm)")
    ax[0].set_xlabel("n")
    ax[0].grid()
    ax[1].plot(n, v[0], label="Old")
    ax[1].loglog(n, v[1], label="New")
    ax[1].legend()
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("construction time [s]")
    ax[1].grid()
    ax[2].loglog(n, v[0] / v[1])
    ax[2].set_xlabel("n")
    ax[2].set_ylabel("speedup")
    ax[2].grid()
    fg.savefig(f"{k.upper()}.png")

plt.tight_layout()
plt.show()
