import argparse
import datetime
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# checked against https://gml.noaa.gov/grad/solcalc/calcdetails.html

args = argparse.ArgumentParser()
args.add_argument('--longitude', type=float, default=-0.220310)
args.add_argument('--latitude', type=float, default=51.413370)

args.add_argument('--year', type=int, default=2012)
args.add_argument('--month', type=int, default=11)
args.add_argument('--date', type=int, default=7)

args.add_argument('--hour', type=int, default=0)
args.add_argument('--minute', type=int, default=0)
args.add_argument('--second', type=int, default=0)

args = args.parse_args()


def date2julian(observer_datetime):
    day_fraction = observer_datetime.hour / 24.0 + observer_datetime.minute / 1440.0 + observer_datetime.second / 86400.0

    julian_date = observer_datetime.toordinal() + 1721424.5 + day_fraction

    return julian_date


def date2angles(observer_datetime, observer_latitude, observer_longitude):
    # https://en.wikipedia.org/wiki/Position_of_the_Sun

    julian_date = date2julian(observer_datetime)
    julian_century = (julian_date - 2451545) / 36525

    eccent_earth_orbit = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)

    num_days_since_greenwich_noon = julian_date - 2451545.0

    mean_longitude = 280.460 + 0.9856474 * num_days_since_greenwich_noon
    mean_anomaly = 357.528 + 0.9856003 * num_days_since_greenwich_noon

    mean_longitude = mean_longitude % 360
    mean_anomaly = mean_anomaly % 360

    ecliptic_longitude = mean_longitude + 1.915 * math.sin(math.radians(mean_anomaly)) + 0.020 * math.sin(
        2 * math.radians(mean_anomaly))

    ecliptic_latitude = 0.0

    distance_from_sun2earth = 1.00014 - 0.01671 * math.cos(mean_anomaly) - 0.00014 * math.cos(2 * mean_anomaly)

    obliquity_of_the_ecliptic = 23.439 - 0.0000004 * num_days_since_greenwich_noon

    # obliquity_of_the_ecliptic = math.radians(obliquity_of_the_ecliptic)

    right_ascension = math.atan2(math.cos(math.radians(ecliptic_longitude)),
                                 math.cos(math.radians(obliquity_of_the_ecliptic)) * math.sin(
                                     math.radians(ecliptic_longitude))
                                 )

    declination = math.asin(
        math.sin(math.radians(obliquity_of_the_ecliptic)) * math.sin(math.radians(ecliptic_longitude)))

    right_ascension, declination = math.degrees(right_ascension), math.degrees(declination)

    var_y = math.tan(math.radians(obliquity_of_the_ecliptic / 2)) * math.tan(
        math.radians(obliquity_of_the_ecliptic / 2))

    eq_of_time_mins = 4 * math.degrees(var_y * math.sin(2 * math.radians(mean_longitude)) -
                                       2 * eccent_earth_orbit * math.sin(math.radians(mean_anomaly)) +
                                       4 * eccent_earth_orbit * var_y * math.sin(math.radians(mean_anomaly)) * math.cos(
        2 * math.radians(mean_longitude)) -
                                       0.5 * var_y ** 2 * math.sin(4 * math.radians(mean_longitude)) -
                                       1.25 * eccent_earth_orbit ** 2 * math.sin(2 * math.radians(mean_anomaly))
                                       )

    ha_sunrise_deg = math.degrees(
        math.acos(
            math.cos(math.radians(90.833)) /
            (math.cos(math.radians(observer_latitude)) * math.cos(math.radians(declination))) -
            math.tan(math.radians(observer_latitude)) * math.tan(math.radians(declination))
        )
    )

    solar_noon = (720 - 4 * observer_longitude - eq_of_time_mins) / 1440

    sunrise_time = (solar_noon - ha_sunrise_deg * 4 / 1440) * 24

    sunset_time = (solar_noon + ha_sunrise_deg * 4 / 1440) * 24

    return right_ascension, declination, sunrise_time, sunset_time


def sun_position(observer_latitude, observer_longitude, right_ascension, declination, observer_datetime):
    # http://star-www.st-and.ac.uk/~fv/webnotes/chapter7.htm
    # https://github.com/jhaupt/Sidereal-Time-Calculator/blob/master/SiderealTimeCalculator.py
    # https://people.ast.cam.ac.uk/~ioalib/convert.html

    # observer_datetime_utc = observer_datetime - datetime.timedelta(hours=observer_longitude / 15)
    julian_date = date2julian(observer_datetime)

    greenwich_mean_sidereal_time = 18.697374558 + 24.06570982441908 * (julian_date - 2451545)
    greenwich_mean_sidereal_time = greenwich_mean_sidereal_time % 24

    observer_longitude_hrs = observer_longitude / 15
    local_sidereal_time = greenwich_mean_sidereal_time + observer_longitude_hrs

    # Convert local sidereal time to degrees
    local_hour_angle = local_sidereal_time - right_ascension
    local_hour_angle_deg = local_hour_angle * 15
    local_hour_angle_rad = math.radians(local_hour_angle_deg)

    # Compute altitude (a) using the cosine rule
    sin_altitude = math.sin(math.radians(declination)) * math.sin(math.radians(observer_latitude)) + math.cos(
        math.radians(declination)) * math.cos(
        math.radians(observer_latitude)) * math.cos(local_hour_angle_rad)
    altitude_rad = math.asin(sin_altitude)

    # Compute azimuth (A) using the sine rule and cosine rule
    sin_azimuth = -math.sin(local_hour_angle_rad) * math.cos(math.radians(declination)) / math.cos(altitude_rad)
    cos_azimith = (math.sin(math.radians(declination)) - math.sin(math.radians(observer_latitude)) * sin_altitude) / (
            math.cos(math.radians(observer_latitude)) * math.cos(altitude_rad))
    # cos_azimith = math.sin(declination)*math.cos(latitude_rad) - math.cos(declination) * math.cos(local_hour_angle_rad) * math.sin(latitude_rad)

    if cos_azimith <= 0.0:
        azimuth_rad = math.pi - math.asin(sin_azimuth)
    else:
        if sin_azimuth <= 0.0:
            azimuth_rad = 2 * math.pi + math.asin(sin_azimuth)
        else:
            azimuth_rad = math.asin(sin_azimuth)

    # Convert azimuth to degrees
    azimuth_deg = math.degrees(azimuth_rad)

    # Convert altitude to degrees
    altitude_deg = math.degrees(altitude_rad)

    return (azimuth_deg, altitude_deg)

def simulate_sun_path(latitude, longitude, year, month, date):
    observer_datetime = datetime.datetime(year, month, date, 12, 0, 0)
    right_ascension, declination, sunrise_time, sunset_time = date2angles(observer_datetime, latitude, longitude)

    if month > 3 and month < 11:
        sunrise_time += 1
        sunset_time += 1

    azimuth_altitude_list = []

    step_size = 60

    for minute in tqdm(range(int(sunrise_time * 60) + step_size, int(sunset_time * 60) - step_size, step_size)):
        hour = minute / 60
        observer_datetime = datetime.datetime(year, month, date, int(hour), minute % 60, 0)
        right_ascension, declination, _, _ = date2angles(observer_datetime, latitude, longitude)
        azimuth, altitude = sun_position(latitude, longitude, right_ascension, declination, observer_datetime)

        time = observer_datetime.strftime("%H:%M")
        azimuth_altitude_list.append((azimuth, altitude, time))

    hour = sunset_time
    minute = int(sunset_time * 60)
    observer_datetime = datetime.datetime(year, month, date, int(hour), minute % 60, 0)
    right_ascension, declination, _, _ = date2angles(observer_datetime, latitude, longitude)
    azimuth, altitude = sun_position(latitude, longitude, right_ascension, declination, observer_datetime)

    time = observer_datetime.strftime("%H:%M")
    azimuth_altitude_list.append((azimuth, altitude, time))

    return azimuth_altitude_list

def plot_sphere(fig, azimuth_altitude_points, latitude, longitude, year, month, date, pdf_pages):


    # Convert azimuth and altitude to spherical coordinates
    azimuths, altitudes, times = zip(*azimuth_altitude_points)

    theta = np.radians(azimuths)
    phi = np.radians(90 - np.array(altitudes))

    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    radius = 0.1

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    # Define the center of the sphere
    center = [0.8,0.8,0.8]

    ax.plot_surface(x_sphere+center[0], y_sphere+center[1], z_sphere+center[2], color='c', alpha=0.2, label='Sphere Surface')

    # Define azimuth angles for compass directions in degrees
    azimuth_compass = {'N': 0, 'E': 90, 'S': 180, 'W': 270}

    # Radius at which compass labels are placed around the sphere
    compass_label_radius = radius * 2 # Increase a bit if needed
    compass_arrow_radius = radius * 1.7 # Increase a bit if needed


    # Annotate compass directions with arrows
    for direction, angle in azimuth_compass.items():
        # Convert angular coordinates to Cartesian coordinates for the direction
        theta = np.radians(angle)
        phi = np.radians(90)  # Azimuth plane

        x_dir = compass_label_radius * np.sin(phi) * np.cos(theta)
        y_dir = compass_label_radius * np.sin(phi) * np.sin(theta)
        z_dir = compass_label_radius * np.cos(phi)

        ax.text(x_dir+center[0], y_dir+center[1], z_dir+center[2], direction, fontsize=6, ha='center', va='center', color='black')

        x_dir = compass_arrow_radius * np.sin(phi) * np.cos(theta)
        y_dir = compass_arrow_radius * np.sin(phi) * np.sin(theta)
        z_dir = compass_arrow_radius * np.cos(phi)

        # Draw arrow from center to the direction label
        ax.quiver(center[0], center[1], center[2], x_dir, y_dir, z_dir,
                  arrow_length_ratio=0.25, color='black')


    # Plot points on the sphere
    ax.scatter(x, y, z, color='orange', marker='o', s=300, alpha=0.5)

    for i in range(len(x)):
        if i == 0:
            ax.text(x[i], y[i], z[i], f"Sunrise: {times[i]}", horizontalalignment='center', verticalalignment='top', zorder=10)
        elif i == len(x) - 1:
            ax.text(x[i], y[i], z[i], f"Sunset: {times[i]}", horizontalalignment='center', verticalalignment='top', zorder=10)
        else:
            ax.text(x[i], y[i], z[i], times[i], size=5, horizontalalignment='center', verticalalignment='center')

    # Set the view parameters to show the plot from the perspective of the origin
    ax.view_init(elev=10, azim=75)

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)


    months_map = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

    plt.title(f"Sun path for: longitude = {longitude} and latitude = {latitude}\non {date} {months_map[month]} {year}")

    pdf_pages.savefig(fig)

from matplotlib.backends.backend_pdf import PdfPages

# Create a PdfPages object to save the figures to a single PDF file
with PdfPages('35_Castelnau_Report.pdf') as pdf_pages:
    for month in range(1, 12 + 1):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        azimuth_altitude_list = simulate_sun_path(args.latitude, args.longitude, args.year, month, 1)
        plot_sphere(fig, azimuth_altitude_list, args.latitude, args.longitude, year=args.year, month=month, date=1, pdf_pages=pdf_pages)
