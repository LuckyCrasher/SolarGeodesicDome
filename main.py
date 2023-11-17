import numpy as np
from geodesicDome.geodesic_dome import GeodesicDome
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


class SolarPanel:
    _triangle: np.array

    def __init__(self, triangle: np.array):
        self._triangle = triangle
        self._face_color = 1

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

    cmap = plt.get_cmap('viridis')  # You can choose a different colormap if you prefer

    # Plot each triangle with its corresponding color
    for i, panel in enumerate(panels):
        vertices = np.array(panel.get_triangle())
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        color = cmap(panel.get_face_color())  # You can replace 'viridis' with any colormap
        ax.plot_trisurf(x, y, z, color=color)

    plt.show()


def main():
    dome = GeodesicDome(freq=1)
    vert = dome.get_vertices()
    trig = dome.get_triangles()

    panels = []
    triangles = vert[trig]
    for triangle in triangles:
        panels.append(SolarPanel(triangle))

    plot_panels(panels[0:len(panels)])


if __name__ == "__main__":
    main()
