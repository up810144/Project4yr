import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.modeling import models
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_continuum
from astropy import units as u
from os import path
from scipy.signal import find_peaks
import scipy as sp

file = '/home/zak/Documents/Project4Yr/firefly_release_0_1_1/example_data/spectra/0266/spec-0266-51630-0623.fits'
hdu = fits.open(file)
t = hdu['COADD'].data
z = hdu[2].data['Z']   # SDSS redshift estimate
print(z)

# Only use the wavelength range in common between galaxy and stellar library.
mask = (t['loglam'] > np.log10(3540)) & (t['loglam'] < np.log10(7409))
fluxy = t['flux'][mask]
print(len(fluxy))
loglam_gal = t['loglam'][mask]
print(len(loglam_gal))
galaxy = fluxy/np.median(fluxy)
print(len(galaxy))

wavelength = 10**loglam_gal
print(len(wavelength))

#STACKED DATA IMPORT
# input_file = '/home/zak/Documents/ProjectData/DAP_Stacks/z_0.5_0.6/CB_C3_LRG/z0506_CBC3LRG_V3_DAP.fits'
#
# hdu = fits.open(input_file)
#
# wavelength = hdu[1].data
# print(wavelength)
# fluxy = hdu[2].data
# len(fluxy)
# #error = flux*0.1
# error = hdu[3].data

#SPECUTILIS ATTEMPT
spectrum = Spectrum1D(flux=galaxy*(1 * u.erg / u.cm**2 / u.s / u.AA), spectral_axis=wavelength*u.AA)
g1_fit = fit_continuum(spectrum)
y_continuum_fitted = g1_fit(wavelength*u.AA)
print(y_continuum_fitted)
spec_normalized = spectrum / y_continuum_fitted


# plt.subplot(2,1,1)
# plt.plot(wavelength, galaxy, label='Spectra',c='C0')
# plt.plot(wavelength,y_continuum_fitted,label='Continuum Fit',c='C1')
# #plt.plot(wavelength, y_continuum_fitted)
# plt.title('Specutilis Continuum Removal')
# plt.legend()
# plt.grid(True)
# spec_normalized = spectrum / y_continuum_fitted
#
# plt.subplot(2,1,2)
# plt.plot(spec_normalized.spectral_axis, spec_normalized.flux,label='Normalised Spectra',c='indianred')
# plt.grid(True)
# plt.legend()
#
# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# ax1.plot(wavelength, galaxy, label='Spectra',c='C0')
# ax1.plot(wavelength,y_continuum_fitted,label='Continuum Fit',c='C1')
# ax1.set_title('Specutilis Continuum Removal')
# ax1.grid(True)
# ax1.legend()
# ax2.plot(spec_normalized.spectral_axis,spec_normalized.flux,label='Normalised Spectra',c='indianred')
#
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# ax2.grid(True)
# ax2.legend()
# plt.xlabel('Wavelength (Angstrom)')
# #plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)')
# f.add_subplot(111, frame_on=False)
#
# plt.tick_params(labelcolor="none", bottom=False, left=False)
# plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)")




#POLYFIT ATTEMPT
p = np.poly1d(np.polyfit(wavelength,galaxy,8))
elines = galaxy - p(wavelength)
#
# plt.subplot(2,1,1)
# plt.title('Numpy Polyfit Continuum Removal')
# plt.plot(wavelength,galaxy,c='C0',label='Spectra')
# plt.plot(wavelength,p(wavelength),'C1',label='Continuum Fit')
# plt.grid(True)
# plt.legend()
# elines = galaxy - p(wavelength)
# plt.subplot(2,1,2)
# plt.plot(wavelength,elines,c='indianred',label='Normalised Spectra')
# plt.grid(True)
# plt.legend()
#
# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# ax1.plot(wavelength,galaxy,c='C0',label='Spectra')
# ax1.plot(wavelength,p(wavelength),'C1',label='Continuum Fit')
# ax1.set_title('Numpy Polyfit Continuum Removal')
# ax1.grid(True)
# ax1.legend()
# ax2.plot(wavelength,elines,c='indianred',label='Normalised Spectra')
#
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# ax2.grid(True)
# ax2.legend()
# plt.xlabel('Wavelength (Angstrom)')
# #plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)')
# f.add_subplot(111, frame_on=False)
#
# plt.tick_params(labelcolor="none", bottom=False, left=False)
# plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)",labelpad=10)
#
#
#SCIPY FIND PEAKS ATTEMPT
galaxy = fluxy/np.median(fluxy)
#plt.plot(wavelength,galaxy)
peaks = find_peaks(galaxy,prominence = 0.1  )
#print(peaks[0])
#new_wav = np.linspace(min(wavelength),max(wavelength),len(peaks[0]))
#print(fluxy[peaks[0]])
#print(wavelength[peaks[0]])
new_flux = np.interp(wavelength,wavelength[peaks[0]],galaxy[peaks[0]])
norm_flux = galaxy-new_flux

#
# plt.subplot(2,1,1)
# plt.title('Scipy Find_Peaks Continuum Removal')
# plt.plot(wavelength[peaks[0]],galaxy[peaks[0]],c='C0',label='Spectra')
# plt.plot(wavelength,galaxy,c='C1',label='Continuum Fit')
# plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)')
# plt.legend()
# plt.grid(True)
# plt.subplot(2,1,2)
# plt.plot(wavelength,norm_flux,c='indianred',label='Normalised Spectra')
# plt.xlabel('Wavelength (Angstrom)')
# plt.legend()
# plt.grid(True)
#
# midarr = np.zeros(len(wavelength))
# #plt.plot(wavelength,midarr,'--')
#

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(wavelength,galaxy,c='C0',label='Spectra')
ax1.plot(wavelength[peaks[0]],galaxy[peaks[0]],c='C1',label='Continuum Fit')
ax1.set_title('Scipy Find_Peaks Continuum Removal')
ax1.grid(True)
ax1.legend()
ax2.plot(wavelength,norm_flux,c='indianred',label='Normalised Spectra')

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
ax2.grid(True)
ax2.legend()
plt.xlabel('Wavelength (Angstrom)')
#plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)')
f.add_subplot(111, frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)",labelpad=10)
# 
#
# interp = sp.interpolate.UnivariateSpline(wavelength,galaxy, s = 12)
# sub_flux = galaxy - interp(wavelength)
#
#
# plt.plot(wavelength,galaxy, label = 'Spectra')
# plt.plot(wavelength,interp(wavelength), label = 'Continuum')
#
# sub_flux = galaxy - interp(wavelength)
# plt.plot(wavelength,sub_flux, label = 'Features')
# plt.legend()
#
# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# ax1.plot(wavelength,galaxy,c='C0',label='Spectra')
# ax1.plot(wavelength,interp(wavelength),c='C1',label='Continuum Fit')
# ax1.set_title('Scipy Univeriate Spline Continuum Removal')
# ax1.grid(True)
# ax1.legend()
# ax2.plot(wavelength,sub_flux, c='indianred',label='Normalised Spectra')
#
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# ax2.grid(True)
# ax2.legend()
# plt.xlabel('Wavelength (Angstrom)')
# #plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)')
# f.add_subplot(111, frame_on=False)
#
# plt.tick_params(labelcolor="none", bottom=False, left=False)
# plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)",labelpad=10)
#
