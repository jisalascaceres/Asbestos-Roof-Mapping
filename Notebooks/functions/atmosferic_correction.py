import re
import numpy as np
import rasterio
from datetime import datetime
from Py6S import *
import matplotlib.pyplot as plt
from functions.utils import *

   
# main atmospheric correction function
def atmospheric_correction(image_path, imd_path, output_path, wv3_wavelengths = [0.4274, 0.4819, 0.5471, 0.6043, 0.6601, 0.7227, 0.8240, 0.9136], 
                                 Esun = [1757.89,2004.61,1830.18,1712.07,1535.33,1348.08,1055.94,858.77], altitude = 0.02, aot = 0.15,remove_black_pixels=True,
                                 histogram_path=False):
    
    '''
    Function to perform atmospheric correction on a WorldView-3 image using Py6S.
    Parameters:
    - image_path: str, path to the input image (GeoTIFF or TIL)
    - imd_path: str, path to the corresponding IMD file of said image
    - output_path: str, path to save the corrected image
    - wv3_wavelengths: list of float, central wavelengths for each band in microns
    - Esun: list of float, solar exoatmospheric irradiance values for each band
    - altitude: float, altitude of the sensor in km (default 0.02 km)
    - aot: float, aerosol optical thickness at 550 nm (default 0.15)
    - remove_black_pixels: bool, whether to mask black pixels (default True)
    - histogram_path: str or False, if provided, saves histograms of the corrected bands to this path
    Returns:
    - Saves the atmospherically corrected image at output_path
    - if image_path is a TIL file, it first combines the tiles into a single GeoTIFF
    
    
    '''
    # Extract IMD data 
    
    imd_data = parse_imd(imd_path)
    theta_s = 90 - imd_data['sunEl']   # solar zenith
    absCal = imd_data['absCalFactor']
    effBW = imd_data['effectiveBandwidth']
    d = earth_sun_distance(imd_data['acqTime'])
    SatAngle = imd_data['SatAngleView']
    mean_sunAz = imd_data['meanSunAz']
    acq_date = datetime.fromisoformat(imd_data['acqTime'].replace("Z", ""))
    month, day = acq_date.month, acq_date.day
    
    
    
    # Read image 
    
    if image_path.endswith('.TIL'):
        # if the image is a TIL file, first combine the different tiles into a single one.
        tif_path = image_path.replace('.TIL','_Combined.tif')
        combine_TIL(image_path, tif_path)
        image_path = tif_path

        
    with rasterio.open(image_path) as src:
        img = src.read().astype('float32')  # shape: (bands, rows, cols)
        img = np.ma.masked_equal(img, 0)  # mask no data values (assuming 0 is no data)
        profile = src.profile
        if remove_black_pixels:
            img, profile = mask_black_pixels(img, profile)
    
    print ('Image shape (bands, rows, cols):', img.shape)
    
    assert img.shape[0] == len(wv3_wavelengths), "Number of bands in image does not match number of provided wavelengths"
    assert img.shape[0] == len(Esun), "Number of bands in image does not match number of provided Esun values"
    
    # Apply atmospheric correction band by band
    reflectance_surf = np.zeros_like(img, dtype='float32')
    for i in range(img.shape[0]):
        print(f"Processing band {i+1}...")

        L = dn_to_radiance(img[i], absCal[i], effBW[i])
        rho_toa = radiance_to_reflectance(L, Esun[i], d, theta_s)

        coeffs = py6s_correction_for_band(wavelength_microns=wv3_wavelengths[i],
                                         theta_s=theta_s, month=month, day=day,
                                            solar_az=mean_sunAz, aot=aot,
                                            SatAngle=SatAngle, altitude_km=altitude)
        
        T_total = coeffs['T_total']
        rho_path = coeffs['rho_path']
        S = coeffs['S']

        rho_surf = (rho_toa - rho_path) / (T_total + S * (rho_toa - rho_path))
        
        # normalize
        reflectance_surf[i] = rho_surf
        
        aux = reflectance_surf[i]-np.min(reflectance_surf[i])
        reflectance_surf[i] = aux/(np.max(reflectance_surf[i])-np.min(reflectance_surf[i]))
        
    # Save the corrected image
    profile.update(dtype='float32', count=reflectance_surf.shape[0])
    
    if histogram_path:
        build_histograms(reflectance_surf, bins=100, save_path=histogram_path)
        print (f"\n Histograms saved in path: {histogram_path}") 
        
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(reflectance_surf)

    print(f"\n Correction Complete. Saved in path: {output_path}")
    
    
    