import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

from GeodesicDome2 import BetterGeodesicDomeGenerator


def map_range(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


def sun_direction(time_seconds, sun):
    # Constants
    seconds_in_a_day = 86400  # Number of seconds in a day

    # Calculate the angle based on time
    angle = 2 * np.pi * (time_seconds % seconds_in_a_day) / seconds_in_a_day

    # Calculate the elevation angle (in radians) based on the latitude and time
    days_in_year = 365.25
    declination = 0.409 * np.sin(2 * np.pi * (days_in_year - 81) / 365)  # Approximate declination
    hour_angle = 2 * np.pi * (time_seconds % seconds_in_a_day) / seconds_in_a_day - np.pi  # Adjust for solar noon
    elevation_angle = np.arcsin(np.sin(sun.latitude) * np.sin(declination) +
                                np.cos(sun.latitude) * np.cos(declination) * np.cos(hour_angle))

    # Calculate the spherical coordinates
    x = np.cos(elevation_angle) * np.cos(angle)
    y = np.cos(elevation_angle) * np.sin(angle)
    z = np.sin(elevation_angle)

    # Normalize the vector
    sun_vector = np.array([x, y, z])
    sun_vector /= np.linalg.norm(sun_vector)

    return sun_vector


def seconds_to_hms(time_seconds):
    # Calculate hours, minutes, and seconds
    hours = time_seconds // 3600
    remaining_seconds = time_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    # Format the time as HH:MM:SS
    time_string = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return time_string


def find_sunrise_sunset(sun):
    # Constants
    seconds_in_a_day = 86400  # Number of seconds in a day

    # Iterate over time to find sunrise and sunset
    for time_seconds in range(seconds_in_a_day):
        sun_vector = sun_direction(time_seconds, sun)

        # Check for sunrise (sun crosses horizon going upwards)
        if sun_direction(time_seconds - 1, sun)[2] <= 0 < sun_vector[2]:
            sunrise_time = time_seconds
            break

    # Iterate over time to find sunset (sun crosses horizon going downwards)
    for time_seconds in range(sunrise_time, seconds_in_a_day):
        sun_vector = sun_direction(time_seconds, sun)

        if sun_direction(time_seconds - 1, sun)[2] >= 0 > sun_vector[2]:
            sunset_time = time_seconds
            break
    return sunrise_time, sunset_time


def iterate_sunrise_to_sunset(sun, interval=600):
    # Get sunrise and sunset times
    sunrise_time, sunset_time = find_sunrise_sunset(sun)
    print(f"Sunrise: {seconds_to_hms(sunrise_time)}")
    print(f"Sunset: {seconds_to_hms(sunset_time)}")
    print(f"Daytime: {sunset_time-sunrise_time}s")

    # Iterate over the time range from sunrise to sunset with the specified interval
    for time_seconds in range(sunrise_time, sunset_time, interval):
        yield time_seconds


class Sun:

    def __init__(self, latitude, longitude, time=0):
        self.latitude = latitude
        self.longitude = longitude
        self.current_time = time
        self.direction = sun_direction(self.current_time, self)

    def update_sun_direction(self, new_time):
        self.current_time = new_time
        self.direction = sun_direction(self.current_time, self)

    def get_direction(self):
        return -1*self.direction

    def get_time_str(self):
        return seconds_to_hms(self.current_time)


class GeodesicDome:

    def __init__(self, sun, radius=1, subdivisions=0, center=(0, 0, 0)):
        self.sun = sun
        self.radius = radius
        self.subdivisions = subdivisions
        self.center = center
        self.dome_generator = BetterGeodesicDomeGenerator(radius=radius, subdivisions=subdivisions, center=center)
        self.triangles = self.dome_generator.vertices[self.dome_generator.faces]
        self.panels = []
        for triangle in self.triangles:
            self.panels.append(SolarPanel(triangle, sun))

    def update_shading(self):
        for panel in self.panels:
            panel.update_shading(self.sun)

    def compute_absorbed_power(self):
        for panel in self.panels:
            panel.compute_absorbed_power(self.sun)

    def render_absorbed_power(self):
        maximum_absorbed = 0
        for panel in self.panels:
            if panel.absorbed_power > maximum_absorbed:
                maximum_absorbed = panel.absorbed_power

        for panel in self.panels:
            panel.update_face_color(map_range(panel.absorbed_power, 0, maximum_absorbed, 0, 1))


class SolarPanel:
    _triangle: np.array

    def __init__(self, triangle: np.array, sun: Sun):
        self._triangle = triangle
        self.normal = self.compute_normal()
        self.normalized_normal = self.normal / np.linalg.norm(self.normal)
        self._face_color = self.compute_shading(sun)
        self.center = self.calculate_center_point()
        self.absorbed_power = 0
        self.area = self.get_area()

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
        mag_sun_normal = np.linalg.norm(sun.get_direction())
        #print(f"Magnitudes {mag_panel_normal}, {mag_sun_normal}")
        a = np.dot(self.normalized_normal, sun.get_direction())
        b = np.dot(mag_panel_normal, mag_sun_normal)
        c = a / b
        shading = map_range(c, -1, 1, 0, 1)
        return shading

    def update_shading(self, sun):
        self._face_color = self.compute_shading(sun)

    def compute_absorbed_power(self, sun):
        self.absorbed_power += self.compute_shading(sun) * self.area

    def update_face_color(self, value):
        self._face_color = value

    def calculate_center_point(self):
        x_c = (self._triangle[0][0] + self._triangle[1][0] + self._triangle[2][0]) / 3
        y_c = (self._triangle[0][1] + self._triangle[1][1] + self._triangle[2][1]) / 3
        z_c = (self._triangle[0][2] + self._triangle[1][2] + self._triangle[2][2]) / 3
        return np.array([x_c, y_c, z_c])

    def __repr__(self):
        return f"{self._triangle}"


def show_dome(dome, sun, fig):
    fig.suptitle(sun.get_time_str())
    arrangement = (2, 2)
    show_view(dome, sun, fig, arrangement, 1, elev=90, azim=0)
    show_view(dome, sun, fig, arrangement, 2, elev=10, azim=135)
    show_view(dome, sun, fig, arrangement, 3, elev=30, azim=-90)
    show_view(dome, sun, fig, arrangement, 4, elev=0, azim=180)
    plt.pause(0.005)


def show_view(dome, sun, fig, plot_arrangement, plot_position, elev=0, azim=0):
    size = dome.radius

    ax = fig.add_subplot(plot_arrangement[0], plot_arrangement[1], plot_position, projection="3d")
    ax.set_xlim3d(-size, size)
    ax.set_ylim3d(-size, size)
    ax.set_zlim3d(0, 2 * size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot each triangle with its corresponding color
    for i, panel in enumerate(dome.panels):
        color = plt.cm.Greys(panel.get_face_color())
        poly = Poly3DCollection([panel.get_triangle()], cmap='Greys',
                                facecolors=color,
                                edgecolors='black', linewidths=0.1)
        # ax.quiver(panel.center[0], panel.center[1], panel.center[2],
        #          panel.normal[0], panel.normal[1], panel.normal[2],
        #          color='g', arrow_length_ratio=0.1)
        ax.add_collection3d(poly)

    sun_direction_vector = sun.get_direction()
    sun_start = sun_direction_vector*-(size*1.5)
    ax.quiver(sun_start[0], sun_start[1], sun_start[2],
              sun_direction_vector[0]*(size/3),
              sun_direction_vector[1]*(size/3),
              sun_direction_vector[2]*(size/3),
              color='r', arrow_length_ratio=0.1)

    ax.view_init(elev=elev, azim=azim)
    #ax.grid(False)
    #ax.axis('off')


def run_simulation(dome, sun, fig):
    time_delta = 1000
    for current_time in iterate_sunrise_to_sunset(sun, time_delta):
        if current_time % 100 == 0:
            print(seconds_to_hms(current_time))
        sun.update_sun_direction(current_time)
        dome.update_shading()
        #dome.compute_absorbed_power()
        show_dome(dome, sun, fig)
    #dome.render_absorbed_power()


def save_data(dome):
    panels = dome.panels

    rolling_sum = 0
    for panel in panels:
        rolling_sum += panel.get_area()
    avg_area = rolling_sum / len(panels)
    print(f"Average area = {avg_area}")

    area = []
    for panel in panels:
        area.append(panel.get_area())

    shading = []
    for panel in panels:
        shading.append(panel.get_face_color())

    illumination_area = []
    for panel in panels:
        illumination_area.append(panel.get_area() * panel.get_face_color())

    panel_illumination = pandas.DataFrame()
    panel_illumination['area'] = area
    panel_illumination['shading'] = shading
    panel_illumination['illuminated per area'] = illumination_area
    panel_illumination.to_csv("data/panel_illumination.csv")


def main():
    # 53.338243, -6.215847
    latitude = np.radians(53.338243)  # Replace with the desired latitude in radians
    longitude = np.radians(-6.215847)

    #fig = plt.figure(dpi=1200)
    fig = plt.figure()

    sun = Sun(latitude, longitude)

    radius = 100
    dome = GeodesicDome(sun, radius=radius, subdivisions=3)
    run_simulation(dome, sun, fig)
    plt.show()


if __name__ == "__main__":
    main()
