import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GeodesicDome import GeodesicDome
from GeodesicDome2 import BetterGeodesicDome

matplotlib.use('qtagg')


class SolarPanel:
    _triangle: np.array

    def __init__(self, triangle: np.array):
        self._triangle = triangle
        self._face_color = self.get_area()

    def get_triangle(self):
        return self._triangle

    def get_face_color(self):
        return self._face_color

    def get_area(self):
        # https://www.youtube.com/watch?v=MnpaeFPyn1A&t=86s
        p = self._triangle[0]
        q = self._triangle[1]
        r = self._triangle[2]

        pq = p - q
        pr = p - r
        cross = np.cross(pq, pr)
        mag = np.linalg.norm(cross)
        area = 0.5 * mag
        return area

    def __repr__(self):
        return f"{self._triangle}"


def plot_panels(panels, size=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(-size, size)
    ax.set_ylim3d(-size, size)
    ax.set_zlim3d(-size, size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cmap = plt.get_cmap('viridis')  # You can choose a different colormap if you prefer

    # Plot each triangle with its corresponding color
    for i, panel in enumerate(panels):
        poly = Poly3DCollection([panel.get_triangle()],
                                facecolors=plt.cm.viridis(panel.get_face_color()),
                                edgecolors='black')
        ax.add_collection3d(poly)

    ax.view_init(elev=10, azim=135)
    plt.show()


def main():

    radius = 100
    dome = BetterGeodesicDome(radius=radius, subdivisions=3)

    panels = []
    triangles = dome.vertices[dome.faces]
    for triangle in triangles:
        panels.append(SolarPanel(triangle))

    rolling_sum = 0
    for panel in panels:
        rolling_sum += panel.get_area()
    avg_area = rolling_sum / len(panels)
    print(f"Average area = {avg_area}")

    plot_panels(panels[0:len(panels)], size=radius)


if __name__ == "__main__":
    main()
