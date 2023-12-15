import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from simulation import Simulation, map_range


def show_dome(dome, fig):
    arrangement = (1, 1)
    show_view(dome, fig, arrangement, 1, elev=35, azim=-125)
    #show_view(dome, fig, arrangement, 1, elev=90, azim=0)
    #show_view(dome, fig, arrangement, 2, elev=10, azim=135)
    #show_view(dome, fig, arrangement, 3, elev=30, azim=-90)
    #show_view(dome, fig, arrangement, 4, elev=0, azim=180)


def show_view(dome, fig, plot_arrangement, plot_position, elev=0, azim=0):
    size = dome.radius

    ax = fig.add_subplot(plot_arrangement[0], plot_arrangement[1], plot_position, projection="3d")
    ax.set_xlim3d(-size, size)
    ax.set_ylim3d(-size, size)
    ax.set_zlim3d(0, 2 * size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_irradiance = dome.get_max_irradiance()
    min_irradiance = dome.get_min_irradiance()
    irradiance_range = max_irradiance - min_irradiance
    viridis = mpl.colormaps['viridis']
    # Plot each triangle with its corresponding color
    for i, panel in enumerate(dome.panels):
        color = viridis(map_range(panel.get_total_irradiance(), min_irradiance, max_irradiance, 0, 1))
        #color = 'white'
        #if panel.get_total_irradiance() > 0:
        #    color = 'lightblue'
        poly = Poly3DCollection([panel.vertices], cmap='viridis',
                                facecolors=color,
                                edgecolors='black',
                                linewidths=1)
        pff = ax.add_collection3d(poly)

    #plt.colorbar(pff, ax=ax)
    #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=viridis), ax=ax)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    #ax.axis('off')


def render_simulation(fig, dome, sun):

    def animate(an_dome):
        start_time = datetime.datetime.now()
        show_dome(an_dome, sun, fig)
        end_time = datetime.datetime.now()
        print(f"Took {end_time - start_time}s")

    #animation = FuncAnimation(fig, func=animate, frames=run_simulation(dome, sun), interval=25)
    # setting up wrtiers object
    writer = writers['ffmpeg']
    writer = writer(fps=15, metadata={'artist': 'Me'}, bitrate=1800)

    #animation.save('data/animation.mp4', writer)


def live_view(dome, sun, fig):
    pass
    #for _ in run_simulation(dome, sun):
    #    show_dome(dome, sun, fig)
    #    plt.pause(0.005)


def main():
    start_timestamp = "2023-01-01 00:00"
    end_timestamp = "2023-12-31 23:59"

    latitude = 53.338243
    longitude = -6.215847

    pvlib_parameters = dict(
        module_parameters=dict(pdc0=10, gamma_pdc=-0.004),
        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3)
    )

    radius = 94.5
    subdivisions = 3

    #fig = plt.figure(dpi=1200)
    fig = plt.figure()

    simulation = Simulation(radius, subdivisions, latitude, longitude, start_timestamp, end_timestamp, pvlib_parameters)
    simulation.run()
    simulation.filter_panels(0.96)
    print(simulation.dome.compute_average_area())
    dome = simulation.get_dome()
    show_dome(dome, fig)
    # simulation.export_irradiance()
    #fig = plt.figure(figsize=(16, 9), dpi=1920 / 16)
    plt.show()


if __name__ == "__main__":
    main()
