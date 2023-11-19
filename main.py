import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from GeodesicDome import GeodesicDome
from GeodesicDome2 import BetterGeodesicDome


def map_range(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


class Sun:

    def __init__(self):
        d = np.array([1, 1, 0])
        self.direction = d / np.linalg.norm(d)


class SolarPanel:
    _triangle: np.array

    def __init__(self, triangle: np.array, sun: Sun):
        self._triangle = triangle
        self.normal = self.compute_normal()
        self.normalized_normal = self.normal / np.linalg.norm(self.normal)
        self._face_color = self.compute_shading(sun)
        self.center = self.calculate_center_point()

    def compute_normal(self):
        p = self._triangle[0]
        q = self._triangle[1]
        r = self._triangle[2]

        pq = p - q
        pr = p - r
        normal = np.cross(pq, pr)
        return normal

    def get_triangle(self):
        return self._triangle

    def get_face_color(self):
        return self._face_color

    def get_area(self):
        # https://www.youtube.com/watch?v=MnpaeFPyn1A&t=86s
        mag = np.linalg.norm(self.normal)
        area = 0.5 * mag
        return area

    def compute_shading(self, sun):
        mag_panel_normal = np.linalg.norm(self.normalized_normal)
        mag_sun_normal = np.linalg.norm(sun.direction)
        print(f"Magnitudes {mag_panel_normal}, {mag_sun_normal}")
        a = np.dot(self.normalized_normal, sun.direction)
        b = np.dot(mag_panel_normal, mag_sun_normal)
        c = a / b
        shading = map_range(c, -1, 1, 0, 1)
        return shading

    def calculate_center_point(self):
        x_c = (self._triangle[0][0] + self._triangle[1][0] + self._triangle[2][0]) / 3
        y_c = (self._triangle[0][1] + self._triangle[1][1] + self._triangle[2][1]) / 3
        z_c = (self._triangle[0][2] + self._triangle[1][2] + self._triangle[2][2]) / 3
        return np.array([x_c, y_c, z_c])

    def __repr__(self):
        return f"{self._triangle}"


def plot_panels(panels, sun, size=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(-size, size)
    ax.set_ylim3d(-size, size)
    ax.set_zlim3d(0, 2*size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # Plot each triangle with its corresponding color
    for i, panel in enumerate(panels):
        color = plt.cm.Greys(panel.get_face_color())
        poly = Poly3DCollection([panel.get_triangle()], cmap='Greys',
                                facecolors=color,
                                edgecolors='black')
        #ax.quiver(panel.center[0], panel.center[1], panel.center[2],
        #          panel.normal[0], panel.normal[1], panel.normal[2],
        #          color='g', arrow_length_ratio=0.1)
        ax.add_collection3d(poly)

    sun_direction = sun.direction
    ax.quiver(0, 0, 14, sun_direction[0]*2, sun_direction[1]*5, sun_direction[2]*5,
              color='r', arrow_length_ratio=0.1)

    #ax.view_init(elev=10, azim=135)
    ax.view_init(elev=90, azim=0)
    #ax.view_init(elev=30, azim=-90)
    #ax.view_init(elev=0, azim=0)
    plt.show()


def main():

    radius = 10
    dome = BetterGeodesicDome(radius=radius, subdivisions=1)

    sun = Sun()

    panels = []
    triangles = dome.vertices[dome.faces]
    for triangle in triangles:
        panels.append(SolarPanel(triangle, sun))

    rolling_sum = 0
    for panel in panels:
        rolling_sum += panel.get_area()
    avg_area = rolling_sum / len(panels)
    print(f"Average area = {avg_area}")

    plot_panels(panels[0:len(panels)], sun, size=radius)


if __name__ == "__main__":
    main()
