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



#importing trial observed data
test_file = '/home/zak/Documents/Project4Yr//firefly_release_0_1_1/example_data/spectra/0266/spec-0266-51630-0623.fits'
hduspec = fits.open(test_file)
t = hduspec['COADD'].data

redshift = hduspec[2].data['Z']
trialflux = t['flux']
loglam_gal = t['loglam']
galaxy = trialflux/np.median(trialflux)
trialwave = 10**loglam_gal
restframe_wavelength = trialwave/(1+redshift)

#--------------------------------------------

#sorting the training data
input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
mask = np.where(wavelength[0] < (np.max(restframe_wavelength)-1)) #creating a mask to cut off the red end to match the observed data
#-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
newmodelwave = wavelength[0][mask]
flux = hdu[1].data['features']

newmodelflux=[]

for i in range(len(flux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    newmodelflux.append(flux[i][mask])

metal = hdu[1].data['Z']
age = hdu[1].data['age']

print(metal)
print(age)
print(np.shape(age))
agemetal = np.stack((age,metal))
print(agemetal)
#splitting the training data
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(newmodelflux,metal,random_state=1)

#--------------------------------------------
#removing the continuum of the observered
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(trialwave)

#converting obsereved flux to the common wavelength newmodelwave
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)


#----------------------------------------------
# training the model for metal


xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg_metal.fit(fluxTrain,metalTrain)
preds = xg_reg_metal.predict(fluxTest)
print(np.max(xg_reg_metal.feature_importances_))

rmse = np.sqrt(mean_squared_error(metalTest, preds))
print("RMSE: %f" % (rmse))


gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
stack =np.stack((newflux,gal))

pred2=xg_reg_metal.predict(stack)
print(pred2)


#---------------------------------------------
#visualisation of the model



imp = xg_reg_metal.feature_importances_
'''
xgb.plot_importance(xg_reg_metal)
plt.show()

xgb.plot_tree(xg_reg_metal,num_trees = 4)
plt.show()

plt.plot(newmodelwave,newflux)
plt.axvline(x=newmodelwave[1174],c = 'k')
plt.axvline(x=newmodelwave[3338], c = 'k')
plt.axvline(x = newmodelwave[1882],c='k')
plt.axvline(x = newmodelwave[1851], c ='k')
plt.show()
'''
x_ticks= np.arange(np.min(newmodelwave),np.max(newmodelwave),200)

plt.scatter(newmodelwave,imp)
plt.xticks(x_ticks,rotation = 70)
plt.xlabel('wavelength (Angstom)')
plt.ylabel('Importance')
plt.show()
#--------------------------------------------------------------------------------------
# training for age

fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(newmodelflux,age,random_state=1)
xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg_age.fit(fluxTrain,ageTrain)


pred3=xg_reg_age.predict(stack) #prediction
print(pred3)

#--------------------------------------------------------------------------------
#metalage
#fluxTrain,fluxTest,agemetalTrain,agemetalTest = train_test_split(newmodelflux,agemetal,random_state=1)





#--------------------------------------------------
#notes

#restframe spec
#
#make new wave array, cut model from redend of models to cover the same upper range as the observed

#resample spec to training set new common wave array
#mask flux of models before training to this new array.
