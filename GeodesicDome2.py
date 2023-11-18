import numpy as np
from numpy.linalg.linalg import norm
from scipy.spatial.transform import Rotation


class BetterGeodesicDome:
    """
    Geodesic Dome of Nv vertices and Nf faces.

    self.v: (Nv, 3) array. list of vertices.
    self.f: (Nf, 3) array. list of vertex indices to define faces.
    """

    def __init__(self, radius=1, center=(0, 0, 0)):
        r = (1.0 + np.sqrt(5.0)) / 2.0
        self.v = np.array([
            [-1.0, r, 0.0],
            [1.0, r, 0.0],
            [-1.0, -r, 0.0],
            [1.0, -r, 0.0],
            [0.0, -1.0, r],
            [0.0, 1.0, r],
            [0.0, -1.0, -r],
            [0.0, 1.0, -r],
            [r, 0.0, -1.0],
            [r, 0.0, 1.0],
            [-r, 0.0, -1.0],
            [-r, 0.0, 1.0],
        ], dtype=float)

        r = Rotation.from_euler('x', 0.55, degrees=False)
        self.v = r.apply(self.v)

        # normalize the radius
        length = norm(self.v, axis=1).reshape((-1, 1))
        self.v = self.v / length * radius + center


        self.f = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [5, 4, 9],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ])

