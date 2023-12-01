import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import FixedMount, Array

from GeodesicDome2 import BetterGeodesicDomeGenerator


def map_range(value, left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)


def compute_angle(a, b):
    a_normalised = a / np.linalg.norm(a)
    mag_a = np.linalg.norm(a_normalised)

    b_normalised = b / np.linalg.norm(b)
    mag_b = np.linalg.norm(b_normalised)

    a = np.dot(a_normalised, b_normalised)
    b = np.dot(mag_a, mag_b)
    angle = math.acos(a / b)
    angle = (angle * 180) / np.pi

    return angle


class DomePanel(FixedMount):

    def __init__(self, vertices, pvlib_parameters):
        self.vertices = vertices
        self.pvlib_parameters = pvlib_parameters
        self.A, self.B, self.C = vertices[0], vertices[1], vertices[2]
        self.height_above_ground = self.compute_height_above_ground()
        self.surface_tilt = self.compute_surface_tilt()
        self.azimuth = self.compute_azimuth()
        self.pvlib_parameters['module_parameters']['area'] = self.compute_area()
        self.total_irradiance = 0
        super().__init__(surface_tilt=self.surface_tilt,
                         surface_azimuth=self.azimuth,
                         racking_model='close_mount',
                         module_height=self.module_height)

    def set_total_irradiance(self, v):
        self.total_irradiance = v

    def get_total_irradiance(self):
        return self.total_irradiance

    def compute_area(self):
        # https://www.youtube.com/watch?v=MnpaeFPyn1A&t=86s
        mag = np.linalg.norm(self.compute_normal())
        area = 0.5 * mag
        return area

    def compute_normal(self):
        p = self.A
        q = self.B
        r = self.C

        pq = p - q
        pr = p - r
        normal = np.cross(pq, pr)
        return normal

    def compute_height_above_ground(self):
        oz = (self.A[2] + self.B[2] + self.C[2]) / 3
        return oz

    def compute_azimuth(self):
        north = np.array([1, 0, 0])
        normal = self.compute_normal()
        z_elim = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        normal_z_elim = np.matmul(normal, z_elim)
        angle = compute_angle(north, normal_z_elim)
        return angle

    def compute_surface_tilt(self):
        normal_upright = np.array([0, 0, 1])
        normal = self.compute_normal()
        angle = compute_angle(normal_upright, normal)
        return angle

    def __repr__(self):
        return f"{self.vertices}"


class GeodesicDome:
    def __init__(self, radius=1.0, subdivisions=0, center=(0, 0, 0)):
        self.radius = radius
        self.subdivisions = subdivisions
        self.center = center
        self.dome_generator = BetterGeodesicDomeGenerator(radius=radius, subdivisions=subdivisions, center=center)
        self.triangles = self.dome_generator.vertices[self.dome_generator.faces]
        self.panels = []
        self.max_irradiance = 0
        self.min_irradiance = 0

    def populate_modules(self, pvlib_parameters):
        arrays = []
        for i, triangle in enumerate(self.triangles):
            panel = DomePanel(triangle, pvlib_parameters)
            arrays.append(Array(panel, name=f"panel {i}", **panel.pvlib_parameters))
            self.panels.append(panel)
        return arrays

    def set_max_irradiance(self, v):
        self.max_irradiance = v

    def set_min_irradiance(self, v):
        self.min_irradiance = v

    def get_max_irradiance(self):
        return self.max_irradiance

    def get_min_irradiance(self):
        return self.min_irradiance


class Simulation:

    def __init__(self, radius, subdivisions, latitude, longitude, start_timestamp, end_timestamp, pvlib_parameters):
        self.radius = radius
        self.subdivisions = subdivisions
        self.dome = GeodesicDome(radius, subdivisions)
        self.location = Location(latitude, longitude)
        self.times = pd.date_range(start_timestamp, end_timestamp, freq='1H', tz="Etc/GMT")
        self.pvlib_parameters = pvlib_parameters
        self.weather = self.location.get_clearsky(self.times)
        self.arrays = self.dome.populate_modules(self.pvlib_parameters)

        self.system = PVSystem(arrays=self.arrays, inverter_parameters=dict(pdc0=10000))
        self.mc = ModelChain(self.system, self.location, aoi_model='physical',
                             spectral_model='no_loss')

    def run(self):
        self.mc.run_model(self.weather)
        # for array, pdc in zip(self.system.arrays, self.mc.results.dc):
        #    pdc.plot(label=f'{array.name}')
        # self.mc.results.ac.plot(label='Inverter')
        # plt.show()

    def export_irradiance(self):
        data = self.mc.results.total_irrad
        for i, panel in enumerate(data):
            panel.to_csv(f"data/panels/{i}.csv")

    def compute_total_irradiance(self):
        data = self.mc.results.total_irrad
        panel_irradiance = []
        for i, panel in enumerate(data):
            irradiance = panel.sum()
            total_irradiance = irradiance['poa_global'] + \
                               irradiance['poa_direct'] + \
                               irradiance['poa_diffuse'] + \
                               irradiance['poa_sky_diffuse'] + \
                               irradiance['poa_ground_diffuse']
            panel_irradiance.append((panel, total_irradiance))
        return panel_irradiance

    def filter_panels(self, percentage):
        panels = self.compute_total_irradiance()
        print(len(panels))
        total_irradiance = sum(v for _, v in panels)
        avg_total_irradiance = total_irradiance / len(panels)
        max_irradiance = max([int(v) for _, v in panels])
        min_irradiance = min([int(v) for _, v in panels])
        self.dome.set_max_irradiance(max_irradiance)
        self.dome.set_min_irradiance(min_irradiance)
        filtered = []
        for panel, dome_panel in zip(panels, self.dome.panels):
            _, irradiance = panel
            if irradiance > avg_total_irradiance * percentage:
                dome_panel.set_total_irradiance(irradiance)
                filtered.append(dome_panel)
        print(len(filtered))

    def get_dome(self):
        return self.dome

##def main():
#    start_timestamp = "2023-01-01 00:00"
#    end_timestamp = "2023-12-31 23:59"
#
#    latitude = 53.338243
#    longitude = -6.215847
#
#    pvlib_parameters = dict(
#        module_parameters=dict(pdc0=10, gamma_pdc=-0.004),
#        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3)
#    )
#
#    radius = 94.5
#    subdivisions = 3
#
#    simulation = Simulation(radius, subdivisions, latitude, longitude, start_timestamp, end_timestamp, pvlib_parameters)
#    simulation.run()
#    simulation.filter_panels(0.99)
#    simulation.show_panels()
#    #simulation.export_irradiance()
