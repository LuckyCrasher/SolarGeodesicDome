import math
from datetime import datetime, timedelta

import numpy as np
import scipy

EarthMeanRadius = 6371.01  # In km
AstronomicalUnit = 149597890  # In km


def seconds_to_hms(time_seconds):
    # Calculate hours, minutes, and seconds
    hours = time_seconds // 3600
    remaining_seconds = time_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    # Format the time as HH:MM:SS
    time_string = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return time_string


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


class Location:
    """
    Holds location data
    Longitude
    Latitude
    """

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude


class Time:

    def __init__(self, year, month, day, hours, minutes, seconds):
        self.time = datetime(year, month, day, hours, minutes, seconds)

    def __repr__(self):
        return f"Time({self.time.strftime('%d/%m/%Y - %H:%M:%S')})"

    def generator(self, delta_seconds=1, max_iterations=None):
        iterations = 0

        while max_iterations is None or iterations < max_iterations:
            self.time += timedelta(seconds=delta_seconds)
            yield self.copy()

            iterations += 1

    def add_seconds(self, seconds_to_add):
        self.time += timedelta(seconds=seconds_to_add)

    def next_day(self, hours=0, minutes=0, seconds=0):
        self.time += timedelta(days=1, hours=hours, minutes=minutes, seconds=seconds)

    def copy(self):
        return Time(self.time.year, self.time.month, self.time.day, self.time.hour, self.time.minute, self.time.second)

    def is_greater(self, other_time):
        return self.time > other_time.time

    def get_hours(self):
        return self.time.hour

    def get_minutes(self):
        return self.time.minute

    def get_seconds(self):
        return self.time.second

    def get_year(self):
        return self.time.year

    def get_month(self):
        return self.time.month

    def get_day(self):
        return self.time.day

#class Time:
#
#    def __init__(self, year, month, day, hours, minutes, seconds):
#        self.year = year
#        self.month = month
#        self.day = day
#        self.hours = hours
#        self.minutes = minutes
#        self.seconds = seconds
#
#    def __repr__(self):
#        return f"Time({self.day}/{self.month}/{self.year} - {self.hours}:{self.minutes}:{self.seconds})"
#
#    def generator(self, delta_seconds=1, max_iterations=None):
#        iterations = 0
#
#        while max_iterations is None or iterations < max_iterations:
#            total_seconds = (
#                    self.seconds
#                    + self.minutes * 60
#                    + self.hours * 3600
#                    + self.day * 24 * 3600
#                    + self.month * 30 * 24 * 3600
#                    + self.year * 365 * 24 * 3600
#            )
#
#            total_seconds += delta_seconds
#
#            years, total_seconds = divmod(total_seconds, 365 * 24 * 3600)
#            months, total_seconds = divmod(total_seconds, 30 * 24 * 3600)
#            days, total_seconds = divmod(total_seconds, 24 * 3600)
#            hours, total_seconds = divmod(total_seconds, 3600)
#            minutes, seconds = divmod(total_seconds, 60)
#
#            self.year = int(years)
#            self.month = int(months)
#            self.day = int(days)
#            self.hours = int(hours)
#            self.minutes = int(minutes)
#            self.seconds = int(seconds)
#
#            yield self.copy()
#
#            iterations += 1
#
#    def add_seconds(self, seconds_to_add):
#        self.seconds += seconds_to_add
#
#        # Update minutes and carry over excess seconds
#        self.minutes += self.seconds // 60
#        self.seconds %= 60
#
#        # Update hours and carry over excess minutes
#        self.hours += self.minutes // 60
#        self.minutes %= 60
#
#        # Update days and carry over excess hours
#        self.day += self.hours // 24
#        self.hours %= 24
#
#        # Update months and carry over excess days
#        # Note: This is a simplified approach; it doesn't handle months of different lengths
#        self.month += self.day // 30
#        self.day %= 30
#
#        # Update years and carry over excess months
#        self.year += self.month // 12
#        self.month %= 12
#
#    def next_day(self, hours=0, minutes=0, seconds=0):
#        initial_day = self.day
#        # Add the specified hours, minutes, and seconds
#        self.add_seconds(hours * 3600 + minutes * 60 + seconds)
#        if self.day == initial_day:
#            self.day += 1
#
#    def copy(self):
#        return Time(self.year, self.month, self.day, self.hours, self.minutes, self.seconds)
#
#    def is_greater(self, other_time):
#        if self.year > other_time.year:
#            return True
#        elif self.year < other_time.year:
#            return False
#
#        if self.month > other_time.month:
#            return True
#        elif self.month < other_time.month:
#            return False
#
#        if self.day > other_time.day:
#            return True
#        elif self.day < other_time.day:
#            return False
#
#        if self.hours > other_time.hours:
#            return True
#        elif self.hours < other_time.hours:
#            return False
#
#        if self.minutes > other_time.minutes:
#            return True
#        elif self.minutes < other_time.minutes:
#            return False
#
#        if self.seconds > other_time.seconds:
#            return True
#        elif self.seconds < other_time.seconds:
#            return False
#
#            # If all components are equal, return False
#        return False


def sunpos_psa(time: Time, location: Location):
    # Sunposition calculates by PSA algorithm

    # Calculate time of the day in UT decimal hours
    decimal_hours = time.get_hours() + (time.get_minutes() + time.get_seconds() / 60.0) / 60.0
    # Calculate current Julian Day
    li_aux1 = (time.get_month() - 14) / 12
    li_aux2 = (1461 * (time.get_year() + 4800 + li_aux1)) / 4 + (367 * (time.get_month() - 2 - 12 * li_aux1)) / 12 - (
            3 * ((time.get_year() + 4900 + li_aux1) / 100)) / 4 + time.get_day() - 32075
    julian_date = li_aux2 - 0.5 + decimal_hours / 24.0
    # Calculate difference between current Julian Day and JD 2451545.0
    elapsed_julian_days = julian_date - 2451545.0

    omega = 2.1429 - 0.0010394594 * elapsed_julian_days
    mean_longitude = 4.8950630 + 0.017202791698 * elapsed_julian_days  # Radians
    mean_anomaly = 6.2400600 + 0.0172019699 * elapsed_julian_days
    ecliptic_longitude = mean_longitude + 0.03341607 * math.sin(mean_anomaly) + 0.00034894 * math.sin(
        2 * mean_anomaly) - 0.0001134 - 0.0000203 * math.sin(omega)
    ecliptic_obliquity = 0.4090928 - 6.2140e-9 * elapsed_julian_days + 0.0000396 * math.cos(omega)

    sin_ecliptic_longitude = np.sin(ecliptic_longitude)
    y = math.cos(ecliptic_obliquity) * sin_ecliptic_longitude
    x = math.cos(ecliptic_longitude)
    right_ascension = math.atan2(y, x)
    if right_ascension < 0.0:
        right_ascension = right_ascension + 2 * scipy.pi
    declination = math.asin(math.sin(ecliptic_obliquity) * sin_ecliptic_longitude)

    rad = (scipy.pi / 180)

    greenwich_mean_sidereal_time = 6.6974243242 + 0.0657098283 * elapsed_julian_days + decimal_hours
    local_mean_sidereal_time = (greenwich_mean_sidereal_time * 15 + location.longitude) * rad
    hour_angle = local_mean_sidereal_time - right_ascension
    latitude_in_radians = location.latitude * rad
    cos_latitude = math.cos(latitude_in_radians)
    sin_latitude = math.sin(latitude_in_radians)
    cos_hour_angle = math.cos(hour_angle)
    zenith_angle = (
        math.acos(cos_latitude * cos_hour_angle * math.cos(declination) + math.sin(declination) * sin_latitude))
    d_y = -math.sin(hour_angle)
    d_x = math.tan(declination) * cos_latitude - sin_latitude * cos_hour_angle
    azimuth = math.atan2(d_y, d_x)
    if azimuth < 0.0:
        azimuth = azimuth + scipy.pi * 2
    azimuth = azimuth / rad
    # Parallax Correction
    parallax = (EarthMeanRadius / AstronomicalUnit) * math.sin(zenith_angle)
    zenith_angle = (zenith_angle + parallax) / rad

    return azimuth, zenith_angle


def sun_position_vector(azimuth, zenith_angle):
    azimuth_rad = math.radians(azimuth)
    zenith_rad = math.radians(zenith_angle)

    x = math.sin(zenith_rad) * math.cos(azimuth_rad)
    y = math.sin(zenith_rad) * math.sin(azimuth_rad)
    z = math.cos(zenith_rad)
    sun_vector = np.array([x, y, z])
    sun_vector = sun_vector / np.linalg.norm(sun_vector)

    return sun_vector


class Sun:
    sunrise_time: Time
    sunset_time: Time
    iteration_time: Time

    def __init__(self, latitude, longitude):
        self.position = Location(latitude, longitude)
        self.current_time = Time(2023, 6, 21, 0, 0, 0)
        self.sun_direction = self.compute_sun_position(self.current_time)
        self.time_delta = 1

    def compute_sunrise_sunset(self):
        self.sunrise_time, self.sunset_time = self.find_sunrise_sunset()
        self.iteration_time = self.sunrise_time.copy()

    def compute_sun_position(self, time):
        azimuth, zenith_angle = sunpos_psa(time, self.position)
        self.sun_direction = sun_position_vector(azimuth, zenith_angle)
        return self.sun_direction

    def get_direction(self):
        return -1 * self.sun_direction

    def find_sunrise_sunset(self):
        # Constants
        tmp_time = self.current_time.copy()

        # Iterate over time to find sunrise and sunset
        az_angle, zen_angle = sunpos_psa(tmp_time, self.position)
        initial_sun_vector = sun_position_vector(az_angle, zen_angle)
        for time in tmp_time.generator():
            az_angle, zen_angle = sunpos_psa(time, self.position)
            sun_vector = sun_position_vector(az_angle, zen_angle)
            # Check for sunrise (sun crosses horizon going upwards)
            if initial_sun_vector[2] <= 0 < sun_vector[2]:
                print(f"sunrise {time}: initial sun_vector {initial_sun_vector} sun_vector {sun_vector}")
                sunrise_time = time.copy()
                break
            initial_sun_vector = sun_vector

        # Iterate over time to find sunset (sun crosses horizon going downwards)
        az_angle, zen_angle = sunpos_psa(tmp_time, self.position)
        initial_sun_vector = sun_position_vector(az_angle, zen_angle)
        for time in tmp_time.generator():
            az_angle, zen_angle = sunpos_psa(time, self.position)
            sun_vector = sun_position_vector(az_angle, zen_angle)

            if initial_sun_vector[2] >= 0 > sun_vector[2]:
                print(f"sunset {time}: initial sun_vector {initial_sun_vector} sun_vector {sun_vector}")
                sunset_time = time.copy()
                break
        return sunrise_time, sunset_time

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration_time.is_greater(self.sunset_time):
            raise StopIteration
        else:
            current_time = self.iteration_time.copy()
            self.iteration_time.add_seconds(self.time_delta)  # Move to the next iteration
            self.compute_sun_position(current_time)
            return current_time

    def iterate_sunrise_to_sunset(self, time=None, time_delta=1):
        self.current_time = time if time is not None else self.current_time
        self.compute_sunrise_sunset()
        self.time_delta = time_delta
        return self  # Return the iterator itself
