from sklearn.model_selection import train_test_split
import xgboost as xgb
from astropy.io import fits
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import pylab
input_file='/home/zak/Documents/Project4Yr/MaStarTrainingTest.fits'
hdu = fits.open(input_file)
wavelength = hdu[1].data['wave']
flux = hdu[1].data['features']
metal = hdu[1].data['Z']
hdu.info()
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(flux,metal,random_state=1)
print(np.shape(metalTrain))
print(np.shape(fluxTrain))
print(np.shape(metalTest))
print(np.shape(fluxTest))
D_train = xgb.DMatrix(fluxTrain, label=metalTrain)
D_test = xgb.DMatrix(fluxTest, label=metalTest)
'''
param = {
    'eta': 0.3,
    'max_depth': 3,
    'objective': 'multi:softmax',
    'num_class': 3}

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)
'''

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(fluxTrain,metalTrain)
preds = xg_reg.predict(fluxTest)
print(np.max(xg_reg.feature_importances_))

rmse = np.sqrt(mean_squared_error(metalTest, preds))
print("RMSE: %f" % (rmse))

test_file = '/home/zak/Documents/Project4Yr/firefly_release_0_1_1/example_data/spectra/0266/spec-0266-51630-0623.fits'
#convert to restframe
#try scikit learn decision trees
#test on manga data
#actually normalise continuum
hdu = fits.open(test_file)
t = hdu['COADD'].data
z = hdu[2].data['Z']
#mask = (t['loglam'] > np.log10(3621)) & (t['loglam'] < np.log10(10352))
fluxy = t['flux']
loglam_gal = t['loglam']
galaxy = fluxy/np.median(fluxy)
print(np.shape(galaxy))
gal = galaxy
print(galaxy)
print(gal)
stack =np.stack((galaxy,gal))
print((stack))
print(stack.dtype)
pred2=xg_reg.predict(stack.T)
dtest = xgb.DMatrix(,galaxy)
print(np.shape(fluxTest))
pr
