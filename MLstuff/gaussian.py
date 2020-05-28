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
obsflux = t['flux']
loglam_gal = t['loglam']
galaxy = obsflux/np.median(obsflux)
trialwave = 10**loglam_gal
restframe_wavelength = trialwave/(1+redshift)
vdisp = hduspec[2].data['vdisp']
vdisp_round = int(np.round(vdisp/5.0)*5.0)# rounding vDisp for the models
r_instrument = np.zeros(len(restframe_wavelength))
invar =t['ivar']
rel_error = invar**(-0.5)/obsflux


#sorting the training data
input_file='/home/zak/Documents/Project4Yr/MaStarTraining198.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
mask = np.where(wavelength[0] < (np.max(restframe_wavelength)-2)) #creating a mask to cut off the red end to match the observed data
#-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
newmodelwave = wavelength[0][mask]
Tflux = hdu[1].data['flux']
r_model = (np.loadtxt(os.path.join(os.environ['FF_DIR'],'MaStar_SSP_v0.1_resolution_lin.txt')))[mask]
wave_instrument = newmodelwave

new_rel_error = sr.spectres(newmodelwave,restframe_wavelength,rel_error)#converting the error array to the new wavelength positions

for wi,w in enumerate(newmodelwave):
    r_instrument[wi] = 1900
newmodelflux=[]

for i in range(len(Tflux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    fluxmask=Tflux[i]
    mf = downgrade_MaStar(newmodelwave,fluxmask[mask],r_model[:,1],vdisp_round, wave_instrument, r_instrument)
    newmodelflux.append(mf)


metal = hdu[1].data['Z']
age = hdu[1].data['age']

fluxext=[] #extended fluxext
metalext=[] #extended metallicity
ageext=[] #extended age

for p in range(len(newmodelflux[:])):
    standdev = new_rel_error*newmodelflux[p]
    currentage = age[p]
    currentZ = metal[p]
    for q in range(60):
        fluxext.append(np.random.normal(newmodelflux[p],standdev))
        metalext.append(currentZ)
        ageext.append(currentage)



print(newmodelwave)

print(np.shape(fluxext))
model_removed = []
for y in range(len(fluxext[:])):
    flux_normal = fluxext[y]/np.median(fluxext[y])


    interp = sp.interpolate.UnivariateSpline(newmodelwave,flux_normal, s =20)
    sub_model = flux_normal - interp(newmodelwave)
    model_removed.append(sub_model)

interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(restframe_wavelength)
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)

# plt.plot(newmodelwave,model_removed[9600], c='C1',label='Perturbed')
# plt.plot(newmodelwave,newflux,c='C0',label = 'Observed')
# plt.xlabel('Wavelength (Angstrom)', fontsize = 15)
# plt.ylabel('Flux (1e-17erg/cm^2/s/Ang)', fontsize = 15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend()
# plt.title('Pertubed Model vs Observed Spectra',fontsize = 20)

#splitting the training data
print(np.shape(model_removed))

xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = 60)
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(model_removed,metalext,random_state=1,test_size=0.05)
xg_reg_metal.fit(fluxTrain,metalTrain)

xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = 60)
fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(model_removed,ageext,random_state=1,test_size=0.05)
xg_reg_age.fit(fluxTrain,ageTrain)



#--------------------------------------------
#removing the continuum of the observered
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(restframe_wavelength)
#converting obsereved flux to the common wavelength newmodelwave
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)


gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
stack =np.stack((newflux,gal))

Mpred=xg_reg_metal.predict(fluxTest)

Apred = xg_reg_age.predict(fluxTest)
#---------------------------------------------
#visualisation of the model
#---------------------------------------------
Merror_arr = metalTest - Mpred
Aerror_arr = ageTest - Apred

print(len(Merror_arr))

plt.hist(Merror_arr, bins = 40, rwidth = 0.8)

'''
plt.subplot(2,1,1)
x = np.linspace(-3, 1, 100)
y = x
plt.plot(x,y,c='k',linestyle='--',label='ML = True')
for i in range(len(metalTest)):
    plt.scatter(metalTest[i],Mpred[i],alpha=0.4)
plt.legend(loc='center left', bbox_to_anchor=(1, 0))
plt.ylabel('Derived Metallicity (Dex)', fontsize = 15)
plt.xlabel('True Metallicity (Dex)', fontsize = 15)
plt.title('Machine Learning vs Test Set Metallicity Values', fontsize = 15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-2.5,0.62)

plt.subplot(2,1,2)
for i in range(len(ageTest)):
    plt.scatter(ageTest[i],Apred[i],alpha=0.4)
m = np.linspace(-1,16,100)
n=m
plt.plot(n,m,c='k',linestyle='--')
plt.title('Machine Learning vs Test Set Age Values', fontsize = 15)
plt.ylabel('Derived Age (Gyr)',fontsize = 15)
plt.xlabel('True Age (Gyr)', fontsize = 15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-1.2,15.5)


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

imp = xg_reg_metal.feature_importances_
impmask = np.where(imp > 0.0004)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('wavelength (Angstom)',fontsize = 25)
ax1.set_ylabel('Flux (1e-17erg/cm^2/s/Ang)', color='C0',fontsize=25)
ax1.plot(newmodelwave, newflux, color='C0')
ax1.tick_params(axis='y', labelcolor='C0',labelsize = 20)
ax1.tick_params(axis='x',labelsize = 20)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Importance (log)', color='C1',fontsize=25)  # we already handled the x-label with ax1
ax2.scatter(newmodelwave[impmask], imp[impmask], color='C1',s=35)
ax2.set_yscale('log')
ax2.set_ylim(10**-5,10**-0.8)
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.tick_params(axis='y', labelcolor='C1',labelsize =20)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
'''
