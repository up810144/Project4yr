#Gaussian
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

#sorting the training data
input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
mask = np.where(wavelength[0] < (np.max(restframe_wavelength)-2)) #creating a mask to cut off the red end to match the observed data
#-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
newmodelwave = wavelength[0][mask]
flux = hdu[1].data['features']
r_model = (np.loadtxt(os.path.join(os.environ['FF_DIR'],'MaStar_SSP_v0.1_resolution_lin.txt')))[mask]
wave_instrument = newmodelwave

newinvar = sr.spectres(newmodelwave,restframe_wavelength,invar)#converting the invar array to the new wavelength positions
standdev = np.sqrt(newinvar**-1)

for wi,w in enumerate(newmodelwave):
    r_instrument[wi] = 1900
newmodelflux=[]

for i in range(len(flux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    fluxmask=flux[i]
    mf = downgrade_MaStar(newmodelwave,fluxmask[mask],r_model[:,1],vdisp_round, wave_instrument, r_instrument)
    newmodelflux.append(mf)


metal = hdu[1].data['Z']
age = hdu[1].data['age']
fluxext=[] #extended fluxext
metalext=[] #extended metallicity
ageext=[] #extended age

for p in range(len(newmodelflux[:])):
    currentage = age[p]
    currentZ = metal[p]
    for q in range(60):
        fluxext.append(np.random.normal(newmodelflux[p],standdev))
        metalext.append(currentZ)
        ageext.append(currentage)







agemetal = np.stack((age,metal))


#splitting the training data



xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = 40)
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(fluxext,metalext,random_state=1)
xg_reg_metal.fit(fluxTrain,metalTrain)

xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = 40)
fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(fluxext,ageext,random_state=1)
xg_reg_age.fit(fluxTrain,ageTrain)



#--------------------------------------------
#removing the continuum of the observered
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(restframe_wavelength)
#converting obsereved flux to the common wavelength newmodelwave
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)


gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
stack =np.stack((newflux,gal))

pred=xg_reg_metal.predict(stack)
print(pred)

pred2 = xg_reg_age.predict(stack)
print(pred2)
#---------------------------------------------
#visualisation of the model
#---------------------------------------------


imp = xg_reg_metal.feature_importances_
impmask = np.where(imp != 0)
print(len(imp[impmask]))

plt.plot(newmodelwave,newflux)

plt.scatter(newmodelwave[impmask],newflux[impmask], c = 'C1', s = 10, zorder=3)
#plt.xticks(x_ticks,rotation = 70)
#plt.yscale('log')
plt.xlabel('wavelength (Angstom)')
plt.ylabel('flux')
#plt.ylim(10**-5,10**0)
plt.title('n_estimators = 40')
plt.show()
