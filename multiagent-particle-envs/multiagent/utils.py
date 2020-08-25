import numpy as np


class Point:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]
        self.z = xy[2]

    def new_point(self, xy):
        # generate a new point which is offset by xy
        return Point([self.x + xy[0], self.y + xy[1], self.z + xy[2]])


def LinePlaneCollision(
        planeNormal,
        planePoint,
        rayDirection,
        rayPoint,
        epsilon=1e-6):
    # borrow from
    # https: // rosettacode.org / wiki /
    # Find_the_intersection_of_a_line_with_a_plane  # Python
    flag_intersection = True
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        flag_intersection = False
        Psi = None
        return flag_intersection, Psi

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return flag_intersection, Psi

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.


def doIntersect(p_pos, cell_pos, window):
    endpoints = np.array(window.endpoints)
    planeNormal = np.array([0, 0, 0])
    if window.orient == 'x':
        dim_normal = 0
        planePoint = np.array([window.axis_pos, np.mean(
            endpoints[0:2]), np.mean(endpoints[2:])])
    elif window.orient == 'y':
        dim_normal = 1
        planePoint = np.array(
            [np.mean(endpoints[0:2]), window.axis_pos, np.mean(endpoints[2:])])
    elif window.orient == 'z':
        dim_normal = 2
        planePoint = np.array(
            [np.mean(endpoints[0:2]), np.mean(endpoints[2:]), window.axis_pos])
    else:
        raise ValueError

    planeNormal[dim_normal] = 1
    # if two points stay in the same side of the plane
    # then the segment will not intersect with the plane
    flag1 = 1 if p_pos[dim_normal] > window.axis_pos else 0
    flag2 = 1 if cell_pos[dim_normal] > window.axis_pos else 0
    if flag1 + flag2 != 1:
        return False

    rayDirection = p_pos - cell_pos
    rayPoint = p_pos
    flag_linePlaneCollision, Psi = LinePlaneCollision(
        planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6)
    if not flag_linePlaneCollision:
        return flag_linePlaneCollision
    else:
        psi_del = np.delete(Psi, dim_normal)
        if (psi_del[0] >= endpoints[0] and
            psi_del[0] <= endpoints[1] and
            psi_del[1] >= endpoints[2] and
                psi_del[1] <= endpoints[3]):
            return True
        else:
            return False
