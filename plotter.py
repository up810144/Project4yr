#plotting script for two figure subplots sharing axis
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np
import scipy as sp
from scipy.signal import find_peaks

file1 = '/home/zak/Documents/Project4Yr/firefly_release_0_1_1/output/spFly-spec-0266-51630-062.fits'
file2 = '/home/zak/Documents/Project4Yr/firefly_release_0_1_1/output/spFly-spec-0266-51630-062v2.fits'


hdu1 = fits.open(file1)
hdu2 = fits.open(file2)
'''
hdu1.info()
hdu1[1].header

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(hdu2[1].data['wavelength'],hdu2[1].data['original_data'],c='C0',label='Spectra')
ax1.plot(hdu2[1].data['wavelength'],hdu2[1].data['firefly_model'],c='C1', label='Firefly M11 Fit')
ax1.set_title('M11 vs MaStar SSP Fit')
plt.ylim(2,60)
ax1.grid(True)
ax1.legend()
ax2.plot(hdu1[1].data['wavelength'],hdu1[1].data['original_data'], c='C0',label='Spectra')
ax2.plot(hdu1[1].data['wavelength'],hdu1[1].data['firefly_model'],c='C1',label='Firefly MaStar Fit')

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
plt.show()

'''
file3 = '/home/zak/Documents/ProjectData/MaStar_SSP_v0.1.fits'

hdu3 = fits.open(file3)
wavelength = hdu3[2].data
print(wavelength)
print(max(wavelength))
t = hdu3[1].data[:,0,0,0]
Z = hdu3[1].data[0,:,0,1]
s = hdu3[1].data[0,0,:,2]
fluxgrid = hdu3[3].data

flux_int = fluxgrid[2,4,3,:]
flux_normal = flux_int/np.median(flux_int)
interp = sp.interpolate.UnivariateSpline(wavelength,flux_normal, s = 70)
sub_flux = flux_normal - interp(wavelength)

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(wavelength,flux_normal,c='C0',label='SSP')
ax1.plot(wavelength,interp(wavelength),c='C1', label='Continuum Fit')
ax1.set_title('Training SSP Continuum Removal')
ax1.grid(True)
ax1.legend()
ax2.plot(wavelength,sub_flux, c='indianred',label='Normalised Spectra')

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
plt.show()
