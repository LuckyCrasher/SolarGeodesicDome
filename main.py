import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from simulation import Simulation, map_range


def show_percent_of_avg(dome, fig, plot_arrangement, plot_position, elev=0, azim=0):
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
    for i, panel in enumerate(dome.panels):
        color = 'white'
        if panel.is_filtered:
            color = 'lightblue'
        poly = Poly3DCollection([panel.vertices], cmap='viridis',
                                facecolors=color,
                                edgecolors='black',
                                linewidths=0.1)
        pff = ax.add_collection3d(poly)

    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)


def show_dome(dome, fig):
    arrangement = (1, 2)
    show_view(dome, fig, arrangement, 1, elev=35, azim=-125)
    show_percent_of_avg(dome, fig, arrangement, 2, elev=90, azim=0)
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
                                linewidths=0.1)
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


def make_irradiance_dataframe(panels):
    irradiance_data = panels
    data_panels = []
    for i, panel in enumerate(irradiance_data):
        panel_irradiance_data = panel.irradiance_data
        sum = panel_irradiance_data.sum(axis=1)
        sum.columns = [f"panel{i}"]
        data_panels.append(sum)
    df_irradiance_data = pd.concat(data_panels, axis=1)
    return df_irradiance_data


def compute_power_generation(panels, avg_panel_size):
    # This is only an estimate based on a lot of averages
    # Transparent panel generates 143W per m²
    return panels.multiply(avg_panel_size*143)


def main():
    start_timestamp = "2023-01-01 00:00"
    end_timestamp = "2023-12-31 23:59"

    # Dublin
    latitude = 53.338243
    longitude = -6.215847

    # Quito
    #latitude  = -0.22
    #longitude = -78.5125

    pvlib_parameters = dict(
        module_parameters=dict(pdc0=10, gamma_pdc=-0.004),
        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3)
    )

    radius = 94.5
    subdivisions = 5

    #fig = plt.figure(dpi=1200)
    fig = plt.figure()

    simulation = Simulation(radius, subdivisions, latitude, longitude, start_timestamp, end_timestamp, pvlib_parameters)
    simulation.run()
    filtered_panels = simulation.filter_panels(0.96)
    #irradiance_data = make_irradiance_dataframe(filtered_panels)
    #daily_averages = irradiance_data.resample('D').mean()
    #max_values = daily_averages.max()
    #daily_percentages = daily_averages.divide(max_values)
    avg_panel_size = simulation.dome.compute_average_area()
    print(avg_panel_size)
    #power_generation = compute_power_generation(daily_percentages, avg_panel_size).sum(axis=1)
    #total_power_generated = power_generation.sum() * 24
    #print(total_power_generated / 365.25)
    #power_generation.plot(legend=False)
    dome = simulation.get_dome()
    show_dome(dome, fig)
    # simulation.export_irradiance()
    #fig = plt.figure(figsize=(16, 9), dpi=1920 / 16)
    plt.show()


if __name__ == "__main__":
    main()
