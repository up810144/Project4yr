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


    interp = sp.interpolate.UnivariateSpline(newmodelwave,flux_normal, s = 20)
    sub_model = flux_normal - interp(newmodelwave)
    model_removed.append(sub_model)

agemetal = np.stack((age,metal))
#splitting the training data
fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(model_removed,metalext,random_state=1)
fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(model_removed,ageext,random_state=1)
#fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(fluxext,metalext,random_state=1)
#fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(fluxext,ageext,random_state=1)


#-1-------------------------------------------
#removing the continuum of the observered
interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
sub_flux = galaxy - interp(restframe_wavelength)
#converting obsereved flux to the common wavelength newmodelwave
newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)


#----------------------------------------------
# training the model for metal

fig, axs = plt.subplots(5,2, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.4, wspace=0.2)
axs = axs.ravel()
est = np.arange(10,110,10)

print(est)
for i in range(10):
    xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = est[i])

    xg_reg_metal.fit(fluxTrain,metalTrain)

    xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 6, alpha=10,  n_estimators = est[i])

    xg_reg_age.fit(fluxTrain,ageTrain)

    gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
    stack =np.stack((newflux,gal))

    pred_metal=xg_reg_metal.predict(stack)
    pred_age = xg_reg_age.predict(stack)



    #---------------------------------------------
    #visualisation of the model



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
    eststr = str(est[i])

    # plt.plot(newmodelwave,newflux)
    #
    # plt.scatter(newmodelwave[impmask],newflux[impmask], c = 'C1', s = 10, zorder=3)
    # #plt.xticks(x_ticks,rotation = 70)
    # #plt.yscale('log')
    # plt.xlabel('wavelength (Angstom)')
    # plt.ylabel('flux')
    # #plt.ylim(10**-5,10**0)
    # plt.title('n_estimators = '+eststr)
    # plt.savefig('n_est_'+eststr+'.png')
    #
    axs[i].plot(newmodelwave,newflux )
    axs[i].scatter(newmodelwave[impmask],newflux[impmask],c = 'C1', s = 5, zorder=3)
    axs[i].set_title('N_estimators '+eststr)
    axs[i].text(6200,-0.25,'Z = '+str(pred_metal[0])+' Age = '+str(pred_age[0]))
    #plt.show()
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)",labelpad=20)
plt.xlabel("Wavelength (Angstrom)")
plt.title('alpha = 10, max depths = 6', pad = 20, fontsize = 16)
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
