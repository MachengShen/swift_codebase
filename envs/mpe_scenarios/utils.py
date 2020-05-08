

class Point:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    def new_point(self, xy):
        #generate a new point which is offset by xy
        return Point([self.x + xy[0], self.y + xy[1]])

def _onSegment(p, q, r):
    # TODO: does this method use the points defining windows?
    # TODO: if this method is also needed by other classes, then do not define as
    # a class method
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False

def _orientation(p, q, r):
    # TODO: does this method use the points defining windows?

    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Colinear orientation
        return 0


# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(_p1, _q1, _p2, _q2):
    # TODO: does this method use the points defining windows?
    """
    p1 = Point(_p1)
    q1 = Point(_q1)
    p2 = Point(_p2)
    q2 = Point(_q2)
    """
    # Find the 4 orientations required for
    # the general and special cases
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True

    # Special Cases
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and _onSegment(p1, p2, q1)):
        return True
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and _onSegment(p1, q2, q1)):
        return True
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and _onSegment(p2, p1, q2)):
        return True
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and _onSegment(p2, q1, q2)):
        return True
    # If none of the cases
    return False

# def intersect(self, line: list) -> bool:
# 	#return if intersect with a list of two np arrays specifying a line
# 	raise NotImplementedError