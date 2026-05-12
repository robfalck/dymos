import numpy as np
from scipy import interpolate

# A set of functions that fit splines to the track.


def get_track_points(track, initial_direction=np.array([1, 0])):
    """
    Place nodes along the track centerline for spline fitting.

    Nodes are placed more densely around corners than on straights.

    Parameters
    ----------
    track : object
        A track description object with segment geometry methods.
    initial_direction : np.ndarray
        The initial heading direction as a unit vector.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates of the track nodes.
    """
    # given a track description, place nodes along the centerlines in order to fit a spline
    # through them. Nodes are denser around corners
    pos = np.array([0, 0])
    direction = initial_direction

    points = [[0, 0]]

    for i in range(len(track.segments)):
        radius = track.get_segment_radius(i)
        length = track.get_segment_length(i)
        if radius == 0:
            # on a straight
            endpoint = pos + direction * length

            for j in range(1, length.astype(int) - 1):
                if j % 5 == 0:
                    points.append(pos + direction * j)

            pos = endpoint
        else:
            # corner
            # length is sweep in radians
            side = track.get_corner_direction(i)
            if side == 0:
                normal = np.array([-direction[1], direction[0]])
            else:
                normal = np.array([direction[1], -direction[0]])

            xc = pos[0] + radius * normal[0]
            yc = pos[1] + radius * normal[1]
            theta_line = np.arctan2(direction[1], direction[0])
            theta_0 = np.arctan2(pos[1] - yc, pos[0] - xc)
            if side == 0:
                theta_end = theta_0 + length
                direction = np.array(
                    [np.cos(theta_line + length), np.sin(theta_line + length)]
                )
            else:
                theta_end = theta_0 - length
                direction = np.array(
                    [np.cos(theta_line - length), np.sin(theta_line - length)]
                )
            theta_vector = np.linspace(theta_0, theta_end, 100)

            x, y = parametric_circle(theta_vector, xc, yc, radius)

            for j in range(len(x)):
                if j % 10 == 0:
                    points.append([x[j], y[j]])

            pos = np.array([x[-1], y[-1]])

    return np.array(points)


def parametric_circle(t, xc, yc, R):
    """
    Compute the (x, y) coordinates of a circle at parameter values t.

    Parameters
    ----------
    t : array_like
        Parameter values (angles in radians).
    xc : float
        x-coordinate of the circle center.
    yc : float
        y-coordinate of the circle center.
    R : float
        Radius of the circle.

    Returns
    -------
    x : np.ndarray
        x-coordinates of the circle.
    y : np.ndarray
        y-coordinates of the circle.
    """
    x = xc + R * np.cos(t)
    y = yc + R * np.sin(t)
    return x, y


def get_spline(points, interval=0.0001, s=0.0):
    """
    Fit a B-spline to the given track points and return spline data.

    Parameters
    ----------
    points : np.ndarray
        Array of (x, y) track node positions to fit.
    interval : float
        Step size for the uniform spline parameter array.
    s : float
        Smoothing factor passed to scipy's splprep.

    Returns
    -------
    finespline : list of np.ndarray
        Densely sampled spline coordinates.
    gates : list of np.ndarray
        Spline coordinates evaluated at the original node parameter values.
    gatesd : list of np.ndarray
        First derivatives of the spline at the original node parameter values.
    curv : np.ndarray
        Signed curvature of the spline at each fine sample point.
    single : list of np.ndarray
        First derivatives of the spline at each fine sample point.
    """
    # this function fits the spline
    tck, u = interpolate.splprep(points.transpose(), s=s, k=5)
    unew = np.arange(0, 1.0, interval)
    finespline = interpolate.splev(unew, tck)

    gates = interpolate.splev(u, tck)
    gatesd = interpolate.splev(u, tck, der=1)

    single = interpolate.splev(unew, tck, der=1)
    double = interpolate.splev(unew, tck, der=2)
    curv = (single[0] * double[1] - single[1] * double[0]) / (
        single[0] ** 2 + single[1] ** 2
    ) ** (3 / 2)

    return finespline, gates, gatesd, curv, single


def get_gate_normals(gates, gatesd):
    """
    Compute the outward and inward normal vectors at each gate location.

    Parameters
    ----------
    gates : list of np.ndarray
        Gate positions as [x_array, y_array].
    gatesd : list of np.ndarray
        First-derivative vectors at each gate as [dx_array, dy_array].

    Returns
    -------
    normals : list of list
        List of [normal_outward, normal_inward] unit vector pairs for each gate.
    """
    normals = []
    for i in range(len(gates[0])):
        der = [gatesd[0][i], gatesd[1][i]]
        mag = np.sqrt(der[0] ** 2 + der[1] ** 2)
        normal1 = [-der[1] / mag, der[0] / mag]
        normal2 = [der[1] / mag, -der[0] / mag]

        normals.append([normal1, normal2])

    return normals


def transform_gates(gates):
    """
    Transform gates from column-array format to row-point format.

    Converts from [[x positions], [y positions]] to [[x0, y0], [x1, y1], ...].

    Parameters
    ----------
    gates : list of np.ndarray
        Gate positions in column format [x_array, y_array].

    Returns
    -------
    newgates : list of list
        Gate positions as a list of [x, y] pairs.
    """
    # transforms from [[x positions],[y positions]] to [[x0, y0],[x1, y1], etc..]
    newgates = []
    for i in range(len(gates[0])):
        newgates.append(([gates[0][i], gates[1][i]]))
    return newgates


def reverse_transform_gates(gates):
    """
    Transform gates from row-point format back to column-array format.

    Converts from [[x0, y0], [x1, y1], ...] to [[x positions], [y positions]].

    Parameters
    ----------
    gates : list of list
        Gate positions as a list of [x, y] pairs.

    Returns
    -------
    newgates : np.ndarray
        Gate positions as a (2, N) array with rows for x and y coordinates.
    """
    # transforms from [[x0, y0],[x1, y1], etc..] to [[x positions],[y positions]]
    newgates = np.zeros((2, len(gates)))
    for i in range(len(gates)):
        newgates[0, i] = gates[i][0]
        newgates[1, i] = gates[i][1]
    return newgates


def set_gate_displacements(gate_displacements, gates, normals):
    """
    Apply lateral displacements to gate positions along their normal directions.

    Does not modify the original gates array; returns a new updated copy.

    Parameters
    ----------
    gate_displacements : array_like
        Signed displacement values for each gate (positive = outward normal direction).
    gates : np.ndarray
        Gate positions in column format [x_array, y_array].
    normals : list of list
        List of [normal_outward, normal_inward] unit vector pairs for each gate.

    Returns
    -------
    newgates : np.ndarray
        Updated gate positions after applying the displacements.
    """
    # does not modify original gates, returns updated version
    newgates = np.copy(gates)
    for i in range(len(gates[0])):
        if i > len(gate_displacements) - 1:
            disp = 0
        else:
            disp = gate_displacements[i]
        # if disp>0:
        normal = normals[i][0]  # always points outwards
        # else:
        # 	normal = normals[i][1] #always points inwards
        newgates[0][i] = newgates[0][i] + disp * normal[0]
        newgates[1][i] = newgates[1][i] + disp * normal[1]
    return newgates
