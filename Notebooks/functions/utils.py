import re
import numpy as np
import rasterio
from datetime import datetime
from Py6S import *
import matplotlib.pyplot as plt

# Some support functions

# Function to parse IMD file and extract necessary parameters
def parse_imd(path):
    with open(path, 'r') as f:
        txt = f.read()
    data = {}
    data['absCalFactor'] = [float(x) for x in re.findall(r'absCalFactor\s*=\s*([0-9eE\+\-\.]+)', txt)]
    data['effectiveBandwidth'] = [float(x) for x in re.findall(r'effectiveBandwidth\s*=\s*([0-9eE\+\-\.]+)', txt)]
    data['sunEl'] = float(re.search(r'meanSunEl\s*=\s*([0-9\.]+)', txt).group(1))
    data['acqTime'] = re.search(r'firstLineTime\s*=\s*([0-9T:\.\-Z]+)', txt).group(1)
    data['meanSunAz'] = float(re.search(r'meanSunAz\s*=\s*([0-9\.]+)', txt).group(1))
    data['SatAngleView'] = float(re.search(r'meanOffNadirViewAngle\s*=\s*([0-9\.]+)', txt).group(1))
    return data

# Function to calculate Earth-Sun distance in AU
def earth_sun_distance(date_str):
    date = datetime.fromisoformat(date_str.replace("Z",""))
    day_of_year = date.timetuple().tm_yday
    d = 1 - 0.01672 * np.cos(np.deg2rad(0.9856 * (day_of_year - 4))) #https://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
    return d

# Functions for radiometric calculations
def dn_to_radiance(band_data, absCalFactor, effectiveBandwidth): # PDF absolute radiometric calibration. Digital globe. No gain no offset, images already radiometrically corrected
    return band_data * (absCalFactor / effectiveBandwidth)

# Function to convert radiance to reflectance
def radiance_to_reflectance(L, E_sun, d, theta_s): #https://ecampusontario.pressbooks.pub/remotesensing/chapter/chapter-3-calculations-of-toa-radiance-and-toa-reflectance/
    return (np.pi * L * d**2) / (E_sun * np.cos(np.deg2rad(theta_s)))

# Function to delete all the black pixels in the image
def mask_black_pixels(img, profile):
    # Create a mask: True where at least one band is nonzero
    mask = np.any(img != 0, axis=0)

    # Optionally set nodata value in the profile
    profile.update(nodata=0)

    # Apply mask: set 0s to np.nan or leave them as nodata
    masked_img = img.astype(float)
    masked_img[:, ~mask] = np.nan  # or 
    #masked_img[:, ~mask] = profile["nodata"]
    
    return masked_img, profile

def py6s_correction_for_band(wavelength_microns, theta_s, month, day,solar_az, aot,
                             SatAngle, altitude_km=0.02,atmospheric_profile =AtmosProfile.MidlatitudeSummer,
                             Aero_profile = AeroProfile.Continental): # 
    s = SixS()
    s.geometry = Geometry.User()
    s.geometry.solar_z = theta_s
    s.geometry.solar_a = solar_az  # mean solar azimut
    s.geometry.view_z = SatAngle  
    s.geometry.month = month
    s.geometry.day = day

    s.atmos_profile = AtmosProfile.PredefinedType(atmospheric_profile)
    s.aero_profile = AeroProfile.PredefinedType(Aero_profile)
    s.aot550 = aot

    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(altitude_km)

    # Set the spectral response as the approximate center band
    s.wavelength = Wavelength(wavelength_microns) 
    s.run()

    return {
        'T_total': s.outputs.transmittance_total_scattering.total,
        'rho_path': s.outputs.apparent_reflectance,
        'S': s.outputs.spherical_albedo.total
    }
    
def combine_TIL(til_path, output_path,):
    from osgeo import gdal

    # Open the .TIL file
    ds = gdal.Open(til_path)

    # Translate to GeoTIFF
    gdal.Translate(output_path, ds, format="GTiff")

    print("Converted .TIL to GeoTIFF at:", output_path)
    
def build_histograms(img, bins=50,save_path=None):
    num_bands = img.shape[0]
    
    # Create a grid of 2 columns if is even or 3 if is odd
    fig, axes = plt.subplots(nrows=(num_bands + 1) // 2, ncols=2, figsize=(12, num_bands * 2))
    
    if num_bands == 1:
        axes = [axes]  # ensure axes is iterable
        
    for i in range(num_bands):
        row = i // 2
        col = i % 2
        ax = axes[row][col] if num_bands > 1 else axes[0]
        
        band_data = img[i].compressed()  # get non-masked data
        ax.hist(band_data, bins=bins, color='blue', alpha=0.7)
        ax.set_title(f'Band {i+1} Histogram')
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Frequency')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()