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



#importing trial data
test_file = '/home/zak/Documents/Project4Yr//firefly_release_0_1_1/example_data/spectra/0266/spec-0266-51630-0623.fits'
hduspec = fits.open(test_file)
t = hduspec['COADD'].data

redshift = hduspec[2].data['Z']
trialflux = t['flux']
print(trialflux)
loglam_gal = t['loglam']
galaxy = trialflux/np.median(trialflux)
trialwave = 10**loglam_gal
restframe_wavelength = trialwave/(1+redshift)
# making the flux arrays the same size
#newflux = sr.spectres(wavelength[1],restframe_wavelength,galaxy)

#sorting the training data
input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']

mask = np.where(wavelength[0] < np.max(restframe_wavelength)) #creating a mask to cut off the red end to match the observed data
newmodelwave = wavelength[0][mask]
print(len(newmodelwave))
print(len(restframe_wavelength))
flux = hdu[1].data['features']

newmodelflux=[]

len(mask)
for i in range(len(flux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

    newmodelflux.append(flux[i][mask])


metal = hdu[1].data['Z']

fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(newmodelflux,metal,random_state=1)



xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(fluxTrain,metalTrain)
preds = xg_reg.predict(fluxTest)
print(np.max(xg_reg.feature_importances_))

rmse = np.sqrt(mean_squared_error(metalTest, preds))
print("RMSE: %f" % (rmse))

#removing the continuum of the observered
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(trialwave)

print(newmodelwave);print(restframe_wavelength)
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)
print(newmodelwave[-1])
print(restframe_wavelength[-1])


'''
test_file = '/home/zak/Documents/Project4Yr/MLstuff/manga-8078-12701-LOGCUBE.fits'
hdu = fits.open(test_file)
wave = hdu[6].data
print(np.max(wave))
z = 0.0267
restframe_wavelength = wave/(1+z)

hdu[0].header
newflux = sp.spectres(newmodelwave,wave,flux) #wave,flux = observed - newwave = model flux cut at red end.

#mask = (restframe_wavelength > 3620) & (restframe_wavelength < 10355) # masking areas the models dont conver

trialflux = hdu[1].data[:,37,37]
galaxy = trialflux/np.median(trialflux)
trialwave = hdu[6].data
'''


#convert to restframe
#try scikit learn decision trees
#test on manga data
#actually normalise continuum


gal = sub_flux

stack =np.stack((sub_flux,gal))
pred2=xg_reg.predict(stack)
print(pred2)


#
#restframe spec
#
#make new wave array, cut model from redend of models to cover the same upper range as the observed

#resample spec to training set new common wave array
#mask flux of models before training to this new array.
