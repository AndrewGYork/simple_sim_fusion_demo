#!/usr/bin/python3
# Dependencies from the scipy stack https://www.scipy.org/stackspec.html :
import numpy as np
from scipy.signal import fftconvolve
# Dependencies from https://github.com/AndrewGYork/tools/blob/master/np_tif.py :
import np_tif

"""
Demonstration of how to use iterative deconvolution to reconstruct
densities from simulated 3D SIM data. To save computation time, we'll
ignore the y-dimension, and simulate x-z data.

This script outputs a bunch of TIF files on disk. Use ImageJ to view
them.
"""

# Define a 2D x-z test object
print("Constructing test object")
n_z, n_x = 60, 60
true_density = np.zeros((n_z, n_x))
true_density[n_z//2+5, n_x//2+1] = 1
true_density[n_z//2-5, n_x//2-1] = 1
true_density[n_z//2, ::4] = 1
true_density[n_z//2-10, ::5] = 1
np_tif.array_to_tif(true_density, '1_true_density.tif')

# Define an x-z emission PSF
print("Constructing emission PSF")
na_limit = 0.25 * np.pi
k_magnitude = 0.15
k_z = np.fft.fftfreq(n_z).reshape(n_z, 1)
k_x = np.fft.fftfreq(n_x).reshape(1, n_x)
k_abs = np.sqrt(k_x**2 + k_z**2)
with np.errstate(divide='ignore', invalid='ignore'): # Ugly divide-by-zero code
    k_theta = np.nan_to_num(np.arccos(k_z / k_abs))
psf_field_ft = np.zeros((n_z, n_x), dtype=np.complex128)
psf_field_ft[(k_theta < na_limit) &      # Limited NA
             (np.abs(k_abs - k_magnitude) < 0.01) # Monochromatic
             ] = 1
np_tif.array_to_tif(np.fft.fftshift(np.abs(psf_field_ft)), '2_psf_field_ft.tif')
psf_field = np.fft.fftn(psf_field_ft)
psf_intensity = np.fft.fftshift(np.abs(psf_field * np.conj(psf_field)))
np_tif.array_to_tif(psf_intensity, '3_psf_intensity.tif')

# Define a set of x-z SIM-like illuminations
print("Constructing SIM illuminations")
pix_shifts = np.arange(0, 10, 2)
illumination_field_ft = np.zeros((n_z, n_x), dtype=np.complex128)
illumination_field_ft[((np.abs(k_theta - na_limit) < 0.01) | # High angle beams
                       (np.abs(k_theta) < 0.01)) &           # Low angle beam
                      (np.abs(k_abs - k_magnitude) < 0.01)   # Monochromatic
                      ] = 1
np_tif.array_to_tif(np.fft.fftshift(np.abs(illumination_field_ft)),
                    '4_illumination_field_ft.tif')
illumination_intensities = []
for ps in pix_shifts:
    phase = np.exp(-2j * np.pi * k_x * ps)
    illumination_field = np.fft.fftn(illumination_field_ft * phase)
    illumination_intensities.append(
        np.fft.fftshift(np.abs(illumination_field *
                               np.conj(illumination_field))))
np_tif.array_to_tif(np.array(illumination_intensities),
                    '5_illumination_intensity.tif')

# Define forward measurement operator H
crop_z, crop_x = 7, 5
def H(density):
    images = np.zeros((len(illumination_intensities),
                       n_z - 2*crop_z,
                       n_x - 2*crop_x))
    for i, illumination in enumerate(illumination_intensities):
        glow = density * illumination
        blurred_glow = fftconvolve(glow, psf_intensity, mode='same')
        blurred_glow[blurred_glow <= 1e-12] = 1e-12 # Avoid true zeros
        cropped_blurred_glow = blurred_glow[crop_z:n_z-crop_z,
                                            crop_x:n_x-crop_x]
        images[i, :, :] = cropped_blurred_glow
    return images
        
# Define H_t, the transpose of the forward measurement operator
def H_t(ratio):
    correction_factor = np.zeros((n_z, n_x))
    for i, illumination in enumerate(illumination_intensities):
        padded_ratio = np.pad(ratio[i, :, :],
                              ((crop_z, crop_z),
                               (crop_x, crop_x)),
                              mode='constant')
        blurred_padded_ratio = fftconvolve(
            padded_ratio, psf_intensity, mode='same')
        correction_factor += illumination * blurred_padded_ratio
    return correction_factor

# Use the forward measurement operator to produce simulated data
print("Constructing simulated measurement data")
noiseless_measurement = H(true_density)
brightness = 3
noisy_measurement = np.random.poisson(brightness * noiseless_measurement) + 1e-9
np_tif.array_to_tif(noisy_measurement, '6_noisy_measurement.tif')

# Use H and H_t to deconvolve the simulated data via Richardson-Lucy
# deconvolution
num_iterations = 2000
H_t_norm = H_t(np.ones_like(noisy_measurement)) # Normalization factor
estimate = np.ones_like(true_density) # Naive initial belief
estimate_history = []
for i in range(num_iterations):
    if i %25 == 0:
        print("Decon iteration", i)
    estimate *= H_t(noisy_measurement / H(estimate)
                    ) / H_t_norm
    estimate_history.append(estimate.copy())
np_tif.array_to_tif(np.stack((true_density, estimate), axis=0),
                    '8_final_estimate.tif',
                    slices=1, channels=2)
np_tif.array_to_tif(np.asarray(estimate_history), '7_estimate_history.tif')

