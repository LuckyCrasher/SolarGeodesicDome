import numpy as np
from numpy.linalg.linalg import norm
from scipy.spatial.transform import Rotation


def generate_vertices_and_faces(radius, center):
    # Original vertices taken from Tom Davis
    # tomrdavis@earthlink.net
    # http://www.geometer.org/mathcircles
    r = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0, r, 0.0],  # 0
        [1.0, r, 0.0],  # 2
        [-1.0, -r, 0.0],  # 3
        [1.0, -r, 0.0],  # 4
        [0.0, -1.0, r],  # 5
        [0.0, 1.0, r],  # 6
        [0.0, -1.0, -r],  # 7
        [0.0, 1.0, -r],  # 8
        [r, 0.0, -1.0],  # 9
        [r, 0.0, 1.0],  # 10
        [-r, 0.0, -1.0],  # 11
        [-r, 0.0, 1.0],  # 12
    ], dtype=float)

    r = Rotation.from_euler('x', 0.55, degrees=False)
    vertices = r.apply(vertices)

    # normalize the radius
    length = norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    vertices = clamp_small_to_zero(vertices)

    print("Rotated vertices")
    print(vertices)

    faces = np.array([
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

    return vertices, faces


def clamp_small_to_zero(values):
    tol = 1e-13
    values[np.abs(values) <= tol] = 0
    return values


def compute_middle_vertex(a, b, radius=1):
    c = a + b
    c_prime = c / norm(c)
    c_prime *= radius
    c_prime = clamp_small_to_zero(c_prime)
    return c_prime


class BetterGeodesicDomeGenerator:

    def __init__(self, subdivisions=0, radius=1.0, center=(0, 0, 0)):
        self.subdivisions = subdivisions
        self.radius = radius
        self.center = center

        self.vertices, self.faces = generate_vertices_and_faces(radius, center)
        self.subdivide(divisions=subdivisions, radius=radius)
        self.split_on_equator()
        print("FINISHED GENERATING")

    def split_on_equator(self):
        tolerance = self.radius/10
        valid_faces = []
        for face in self.faces:
            # Checking for z component below 0 with certain tolerance. 
            if self.vertices[face[0]][2] >= 0 - tolerance and \
                    self.vertices[face[1]][2] >= 0 - tolerance and \
                    self.vertices[face[2]][2] >= 0 - tolerance:
                valid_faces.append(face)
        self.faces = valid_faces

    def subdivide(self, divisions=0, radius=1.0):
        for _ in range(divisions):

            # enumerating each triangle
            new_triangles = []
            new_vertex_id = len(self.vertices)
            for triangle in self.faces:
                edge_to_new_vertex_id = {}
                print(f"Subdividing triangle: {triangle}")
                # each vertex connects two edges of the triangle
                # each edge is split in two
                # enumerating each edge of the triangle like
                # edge (0,1) or (1,2) or (2,0) where (1,0) and (0,1) are the same edge
                # therefore (0, 1) (1, 2) (2, 0)
                for i, j in zip([0, 1, 2], [1, 2, 0]):
                    print(f"Edge: {i},{j}")
                    v_id_a = triangle[i]
                    v_id_b = triangle[j]
                    print(f"Vertex ids A:{v_id_a}, B:{v_id_b}")
                    vertex_a = self.vertices[v_id_a]
                    vertex_b = self.vertices[v_id_b]
                    print(f"Vertex A: {vertex_a}")
                    print(f"Vertex B: {vertex_b}")
                    mid_vertex_ab = compute_middle_vertex(vertex_a, vertex_b, radius=radius)
                    print(f"New middle vertex: {mid_vertex_ab}")
                    # add the newly computed vertex to the list of vertices
                    self.vertices = np.append(self.vertices, [mid_vertex_ab], axis=0)
                    # save the new vertex id based on the edge it was generated from
                    print(f"Saved edge {new_vertex_id} as {(i, j)}")
                    edge_to_new_vertex_id[(i, j)] = new_vertex_id

                    new_vertex_id += 1

                # Each triangle will be subdivided into 4 new triangles
                # Triangles are then formed by on known,
                # old vertex and two newly computed ones
                # The last middle triangle is composed of three newly computed vertices
                triangle_a = [triangle[0], edge_to_new_vertex_id[(0, 1)], edge_to_new_vertex_id[2, 0]]
                triangle_b = [triangle[1], edge_to_new_vertex_id[(1, 2)], edge_to_new_vertex_id[0, 1]]
                triangle_c = [triangle[2], edge_to_new_vertex_id[(2, 0)], edge_to_new_vertex_id[1, 2]]
                triangle_d = [edge_to_new_vertex_id[0, 1], edge_to_new_vertex_id[1, 2], edge_to_new_vertex_id[2, 0]]

                print("New triangles:")
                print(triangle_a)
                print(triangle_b)
                print(triangle_c)
                print(triangle_d)

                new_triangles.append(triangle_a)
                new_triangles.append(triangle_b)
                new_triangles.append(triangle_c)
                new_triangles.append(triangle_d)

                print()
            # override old faces with newly generated ones
            self.faces = np.array(new_triangles)
