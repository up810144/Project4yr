from sklearn.model_selection import train_test_split
import xgboost as xgb
from astropy.io import fits
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as npx
import pylab
import numpy as np
#import spectral_resampling as sp
from scipy.signal import find_peaks
import scipy as sp


input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
flux = hdu[1].data['features']
metal = hdu[1].data['Z']
s
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(flux,metal,random_state=1)



xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(fluxTrain,metalTrain)
preds = xg_reg.predict(fluxTest)
print(np.max(xg_reg.feature_importances_))

rmse = np.sqrt(mean_squared_error(metalTest, preds))
print("RMSE: %f" % (rmse))
'''
#importing trial data
test_file = '/home/zak/Documents/Project4Yr//firefly_release_0_1_1/example_data/spectra/0266/spec-0266-51630-0623.fits'
hdu = fits.open(test_file)
t = hdu['COADD'].data

mask = (t['loglam'] > np.log10(3620)) & (t['loglam'] < np.log10(10355)) # masking areas the models dont conver
redshift = hdu[2].data['Z']
trialflux = t['flux'][mask]
print(trialflux)
loglam_gal = t['loglam'][mask]
galaxy = trialflux/np.median(trialflux)
trialwave = 10**loglam_gal
restframe_wavelength = trialwave/(1+redshift)
# making the flux arrays the same size
newflux = sp.spectres(wavelength[1],restframe_wavelength,galaxy)
'''
test_file = '/home/zak/Documents/Project4Yr/MLstuff/manga-8078-12701-LOGCUBE.fits'
hdu = fits.open(test_file)
mask = (trialwave > 3620) & (trialwave < 10355) # masking areas the models dont conver

trialflux = hdu[1].data[:,37,37][mask]
galaxy = trialflux/np.median(trialflux)
trialwave = hdu[6].data[mask]
interp = sp.interpolate.UnivariateSpline(trialwave,galaxy, s = 70)
sub_flux = galaxy - interp(trialwave)

#convert to restframe
#try scikit learn decision trees
#test on manga data
#actually normalise continuum


gal = sub_flux

stack =np.stack((sub_flux,gal))
pred2=xg_reg.predict(stack)
print(pred2)
