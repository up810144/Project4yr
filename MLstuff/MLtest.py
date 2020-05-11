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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
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
invar =t['ivar'] # error for new training set
rel_error = invar**(-0.5)/trialflux

#--------------------------------------------
#sorting the training data
#--------------------------------------------
input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
mask = np.where(wavelength[0] < (np.max(restframe_wavelength)-2)) #creating a mask to cut off the red end to match the observed data
#-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
newmodelwave = wavelength[0][mask]
flux = hdu[1].data['features']
r_model = (np.loadtxt(os.path.join(os.environ['FF_DIR'],'MaStar_SSP_v0.1_resolution_lin.txt')))[mask]
wave_instrument = newmodelwave
standdev = sr.spectres(newmodelwave,restframe_wavelength,rel_error )#converting the invar array to the new wavelength positions

for wi,w in enumerate(newmodelwave):
	r_instrument[wi] = 1900
newmodelflux=[]
print(len(r_model))
for i in range(len(flux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    fluxmask=flux[i]
    mf = downgrade_MaStar(newmodelwave,fluxmask[mask],r_model[:,1],vdisp_round, wave_instrument, r_instrument)


    newmodelflux.append(mf)

print(np.shape(newmodelflux))
metal = hdu[1].data['Z']
age = hdu[1].data['age']

agemetal = np.stack((age,metal))
fluxext=[] #extended fluxext
metalext=[] #extended metallicity
ageext=[] #extended age

for p in range(len(newmodelflux[:])): #creating a larger training set based off of the error of the input data to give more volume of spectra
	currentage = age[p]
	currentZ = metal[p]
	for q in range(60):
		fluxext.append(np.random.normal(newmodelflux[p],standdev))
		metalext.append(currentZ)
		ageext.append(currentage)
#splitting the training data
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(fluxext,metalext,random_state=1)

#--------------------------------------------
#removing the continuum of the observered
#--------------------------------------------
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(restframe_wavelength)
#plt.plot(restframe_wavelength,sub_flux)

#converting obsereved flux to the common wavelength newmodelwave
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)
newflux_cont = sr.spectres(newmodelwave,restframe_wavelength,galaxy)

plt.subplot(3,1,1)
plt.plot(newmodelwave, newflux, c = 'C0')
plt.title('Observed Flux')
# plt.plot(restframe_wavelength,trialflux,label='Flux',c='C0')
# plt.title('Flux')
plt.subplot(3,1,2)
plt.plot(newmodelwave,fluxext[0],c='C1')
plt.title('Peturbed Flux')
# plt.plot(restframe_wavelength,invar,label='Invar',c='C1')
# plt.title('Invar')
plt.subplot(3,1,3)
plt.plot(newmodelwave,fluxext[0],c='C1')
plt.plot(newmodelwave, newflux, c = 'C0')
plt.title('Comparison')
# plt.plot(restframe_wavelength,rel_error, label='Relative Error',c='C2')
# plt.title('Relative Error')
plt.show()
#----------------------------------------------
# training the model for metal
#----------------------------------------------
'''
xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 5 , alpha=10,  n_estimators = 100)

xg_reg_metal.fit(fluxTrain,metalTrain)
preds = xg_reg_metal.predict(fluxTest)
print(preds)
print(metalTest)
plt.scatter(metalTest,preds)

eval_set = [(fluxTrain, metalTrain), (fluxTest, metalTest)]
eval_metric = ["auc","error"]
%time xg_reg_metal.fit(fluxTrain, metalTrain, eval_metric=eval_metric, eval_set=eval_set, verbose=True)


print((metalTest))
print(len(preds))
print(accuracy_score(metalTest,preds))
print(np.max(xg_reg_metal.feature_importances_))

rmse = np.sqrt(mean_squared_error(metalTest, preds))
print("RMSE: %f" % (rmse))


gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
stack =np.stack((newflux,gal))

pred2=xg_reg_metal.predict(stack)
print(pred2)


#---------------------------------------------
#visualisation of the model
#---------------------------------------------


imp = xg_reg_metal.feature_importances_
impmask = np.where(imp != 0)
print(len(imp[impmask]))

# xgb.plot_importance(xg_reg_metal)
# plt.show()
#
# xgb.plot_tree(xg_reg_metal,num_trees = 4)
# plt.show()
#
# plt.plot(newmodelwave,newflux)
# plt.axvline(x=newmodelwave[1174],c = 'k')
# plt.axvline(x=newmodelwave[3338], c = 'k')
# plt.axvline(x = newmodelwave[1882],c='k')
# plt.axvline(x = newmodelwave[1851], c ='k')
# plt.show()


#wavemask =np.where(np.logical_and(newmodelwave > 7800, newmodelwave < 8200))

#x_ticks= np.arange(np.min(newmodelwave),np.max(newmodelwave),200)

plt.plot(newmodelwave,newflux)

plt.scatter(newmodelwave[impmask],newflux[impmask], c = 'C1', s = 10, zorder=3)
#plt.xticks(x_ticks,rotation = 70)
#plt.yscale('log')
plt.xlabel('wavelength (Angstom)')
plt.ylabel('flux')
#plt.ylim(10**-5,10**0)
plt.title('n_estimators = ')
plt.show()

# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('wavelength (Angstom)')
# ax1.set_ylabel('Flux', color='C0')
# ax1.plot(newmodelwave, newflux, color='C0')
# ax1.tick_params(axis='y', labelcolor='C0')
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('Importance', color='C1')  # we already handled the x-label with ax1
# ax2.scatter(newmodelwave[impmask], imp[impmask], color='C1',s=3)
# ax2.set_yscale('log')
# ax2.set_ylim(10**-5,10**0)
# ax2.set_ylim(ax2.get_ylim()[::-1])
# ax2.tick_params(axis='y', labelcolor='C1')
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

#-----------------------------------------------------------------------------
# training for age
#-----------------------------------------------------------------------------

fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(newmodelflux,age,random_state=1)
xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg_age.fit(fluxTrain,ageTrain)


pred3=xg_reg_age.predict(stack) #prediction
print(pred3)

#-----------------------------------------------------------------------------
#metalage
#-----------------------------------------------------------------------------

fluxTrain,fluxTest,agemetalTrain,agemetalTest = train_test_split(newmodelflux,agemetal.T,random_state=1)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
print(agemetalTrain)
xg_reg.fit(fluxTrain,agemetalTrain)
pred4=xg_reg.predict(stack) #prediction
print(pred4)

#------------------------------------------20--------
#notes

#restframe spec
#
#make new wave array, cut model from redend of models to cover the same upper range as the observed

#resample spec to training set new common wave array
#mask flux of models before training to this new array.


#account for vdisp
'''
