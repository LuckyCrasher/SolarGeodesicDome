import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from GeodesicDome import GeodesicDome
from GeodesicDome2 import BetterGeodesicDome


class SolarPanel:
    _triangle: np.array

    def __init__(self, triangle: np.array):
        self._triangle = triangle
        self._face_color = random.Random().randint(0, 10)/10

    def get_triangle(self):
        return self._triangle

    def get_face_color(self):
        return self._face_color

    def __repr__(self):
        return f"{self._triangle}"


def plot_panels(panels):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

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

    ax.view_init(elev=0, azim=0)
    plt.show()


def main():


    g = GeodesicDome()
    print(g.v)

    dome = BetterGeodesicDome()
    print(dome.v)



    panels = []
    triangles = dome.v[dome.f]
    for triangle in triangles:
        panels.append(SolarPanel(triangle))

    plot_panels(panels[0:len(panels)])


if __name__ == "__main__":
    main()
