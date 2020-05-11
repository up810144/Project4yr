#peturb histogram
from sklearn.model_selection import train_test_split
import xgboost as xgb
from astropy.io import fits
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as npx
import pylab
import numpy as np
import spectral_resampling as sr
from scipy.signal import find_peaks
import scipy as sp
from matplotlib import pyplot as plt
from firefly_instrument import downgrade_MaStar
import os,sys
from os.path import join

#importing trial observed data
test_file = '/home/zak/Documents/ProjectData/SDSS_Sample/spec-0266-51602-0614.fits'
hduspec = fits.open(test_file)
t = hduspec['COADD'].data

redshift = hduspec[2].data['Z']
trialflux = t['flux']
loglam_gal = t['loglam']
galaxy = trialflux/np.median(trialflux)
trialwave = 10**loglam_gal
restframe_wavelength = trialwave/(1+redshift)
vdisp = hduspec[2].data['vdisp']
vdisp_round = int(np.round(vdisp/5.0)*5.0)# rounding vDisp for the models
r_instrument = np.zeros(len(restframe_wavelength))
invar =t['ivar']
rel_error = invar**(-0.5)/trialflux


#sorting the training data
input_file='/home/zak/Documents/Project4Yr/MaStarTraining198.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
mask = np.where(wavelength[0] < (np.max(restframe_wavelength)-2)) #creating a mask to cut off the red end to match the observed data
#-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
newmodelwave = wavelength[0][mask]
Tflux = hdu[1].data['flux']
print(Tflux[1])
r_model = (np.loadtxt(os.path.join(os.environ['FF_DIR'],'MaStar_SSP_v0.1_resolution_lin.txt')))[mask]
wave_instrument = newmodelwave

new_rel_error = sr.spectres(newmodelwave,restframe_wavelength,rel_error)#converting the invar array to the new wavelength positions


for wi,w in enumerate(newmodelwave):
    r_instrument[wi] = 1900
newmodelflux=[]

for i in range(len(Tflux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    fluxmask=Tflux[i]
    mf = downgrade_MaStar(newmodelwave,fluxmask[mask],r_model[:,1],vdisp_round, wave_instrument, r_instrument)
    newmodelflux.append(mf)


print(np.shape(newmodelflux))
metal = hdu[1].data['Z']
age = hdu[1].data['age']


pix1=[]
pix2=[]
pix3=[]
loc=[100,1000,3000] #pix 1 - 3 respectively
print(newmodelflux[10][loc[0]])
 #creating a larger training set based off of the error of the input data to give more volume of spectra
standdev = new_rel_error * newmodelflux[10]
for q in range(1000):

    pix1.append(np.random.normal(newmodelflux[10][loc[0]],standdev[loc[0]]))
    pix2.append(np.random.normal(newmodelflux[10][loc[1]],standdev[loc[1]]))
    pix3.append(np.random.normal(newmodelflux[10][loc[2]],standdev[loc[2]]))
print(pix1[10])
plt.subplot(3,1,1)
plt.title('position 100 of 3423')
plt.hist(pix1,bins=25,rwidth=0.85)
plt.subplot(3,1,2)
plt.title('position 1000 of 3423')
plt.hist(pix2,bins=25,rwidth=0.85)
plt.subplot(3,1,3)
plt.title('position 3000 of 3423')
plt.hist(pix3,bins=25,rwidth=0.85)
plt.show()
