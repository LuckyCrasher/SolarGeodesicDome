import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class GeodesicDome:
    """
    Geodesic Dome of Nv vertices and Nf faces.

    self.v: (Nv, 3) array. list of vertices.
    self.f: (Nf, 3) array. list of vertex indices to define faces.
    """

    def __init__(self):
        ## vertices ##
        p = (1 + np.sqrt(5)) / 2
        a = np.sqrt((3 + 4 * p) / 5) / 2
        b = np.sqrt(p / np.sqrt(5))
        c = np.sqrt(3 / 4 - a ** 2)
        d = np.sqrt(3 / 4 - (b - a) ** 2)
        # icosahedron in (r, theta, z) == cylindrical coordinates
        self.v = np.array([
            [0, 0, (c + d / 2)],
            [b, 2 * 0 * np.pi / 5, d / 2],
            [b, 2 * 1 * np.pi / 5, d / 2],
            [b, 2 * 2 * np.pi / 5, d / 2],
            [b, 2 * 3 * np.pi / 5, d / 2],
            [b, 2 * 4 * np.pi / 5, d / 2],
            [b, (2 * 0 + 1) * np.pi / 5, - d / 2],
            [b, (2 * 1 + 1) * np.pi / 5, - d / 2],
            [b, (2 * 2 + 1) * np.pi / 5, - d / 2],
            [b, (2 * 3 + 1) * np.pi / 5, - d / 2],
            [b, (2 * 4 + 1) * np.pi / 5, - d / 2],
            [0, 0, -(c + d / 2)],
        ])
        # icosahedron in (x, y, z) == Cartesian coordinates

        self.v = np.vstack([
            self.v[:, 0] * np.cos(self.v[:, 1]),
            self.v[:, 0] * np.sin(self.v[:, 1]),
            self.v[:, 2]
        ]).T
        # normalize the radius
        self.v *= (1 / self.v[0, 2])

        # fix super small values to zero
        self.tol = 1e-15
        self.v[np.abs(self.v) < self.tol] = 0

        #above = []
        #for i, vert in enumerate(self.v):
        #    if not (vert[2] < 0):
        #        above.append(vert)
        #self.v = np.array(above)

        ## faces ##
        self.f = np.array([
            [2, 0, 1],
            [3, 0, 2],
            [4, 0, 3],
            [5, 0, 4],
            [1, 0, 5],
            [2, 1, 6],
            [7, 2, 6],
            [3, 2, 7],
            [8, 3, 7],
            [4, 3, 8],
            [9, 4, 8],
            [5, 4, 9],
            [10, 5, 9],
            [6, 1, 10],
            [1, 5, 10],
            [6, 11, 7],
            [7, 11, 8],
            [8, 11, 9],
            [9, 11, 10],
            [10, 11, 6],
        ])

        exists = []
        n_vertices = len(self.v)
        for i, face in enumerate(self.f):
            if face[0] < n_vertices and face[1] < n_vertices and face[2] < n_vertices:
                exists.append(face)
        self.f = exists

    def tessellate(self, iter=1):
        def newvert(v0, v1):
            v = v0 + v1
            v /= np.linalg.norm(v)
            return v

        for _ in range(iter):
            f = self.f
            v = self.v
            v2 = []
            vv2v = {}
            vid = len(v)
            for tri in self.f:
                for i, j in zip([0, 1, 2], [1, 2, 0]):
                    if tri[i] < tri[j]:
                        vv2v[tri[i], tri[j]] = vid
                        vv2v[tri[j], tri[i]] = vid
                        vid += 1
                        new_vertex = newvert(v[tri[i]], v[tri[j]])
                        v2.append(new_vertex)

            v = np.vstack([v, np.array(v2)])

            f2 = []
            k = vv2v.keys()
            for tri in self.f:
                # if tri[0] < 0 or vv2v[tri[0], tri[1]] < 0 or vv2v[tri[2], tri[0]] < 0:
                #    continue
                # if tri[1] < 0 or vv2v[tri[1], tri[2]] < 0 or vv2v[tri[0], tri[1]] < 0:
                #    continue
                # if tri[2] < 0 or vv2v[tri[2], tri[0]] < 0 or vv2v[tri[1], tri[2]] < 0:
                #    continue
                # if vv2v[tri[0], tri[1]] < 0 or vv2v[tri[1], tri[2]] < 0 or vv2v[tri[2], tri[0]] < 0:
                #    continue
                if (tri[0], tri[1]) not in k:
                    continue
                if (tri[0], tri[2]) not in k:
                    continue
                if (tri[1], tri[0]) not in k:
                    continue
                if (tri[1], tri[2]) not in k:
                    continue
                if (tri[2], tri[0]) not in k:
                    continue
                if (tri[2], tri[1]) not in k:
                    continue

                print(tri[0], vv2v[tri[0], tri[1]], vv2v[tri[2], tri[0]])
                print(tri[1], vv2v[tri[1], tri[2]], vv2v[tri[0], tri[1]])
                print(tri[2], vv2v[tri[2], tri[0]], vv2v[tri[1], tri[2]])
                print(vv2v[tri[0], tri[1]], vv2v[tri[1], tri[2]], vv2v[tri[2], tri[0]])
                f2.append([tri[0], vv2v[tri[0], tri[1]], vv2v[tri[2], tri[0]]])
                f2.append([tri[1], vv2v[tri[1], tri[2]], vv2v[tri[0], tri[1]]])
                f2.append([tri[2], vv2v[tri[2], tri[0]], vv2v[tri[1], tri[2]]])
                f2.append([vv2v[tri[0], tri[1]], vv2v[tri[1], tri[2]], vv2v[tri[2], tri[0]]])

            self.v = v
            self.f = np.array(f2)

        self.v[np.abs(self.v) < self.tol] = 0
        return self

    def face_normal(self):
        """
        This function is not needed in most cases, since the vertex position is identical to its normal.
        """
        tri = self.v[self.f]
        n = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
        n /= np.linalg.norm(n, axis=0)
        return n
