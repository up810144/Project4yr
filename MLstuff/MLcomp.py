#ML comparison
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
import MaStar_SSP
import os,sys
from os.path import join
from firefly_instrument import downgrade_MaStar


#id = ['0001','0162','0166','0168','0250','0462','0526','0554','0614','0628'] #ID for the testing data file names
id = ['0162','0614']
#Initialising Dictionaries
ML_metal={}
ML_age={}
FF_metalLW={} #firefly lightweighted metalicity
FF_metalMW={} #firefly mass weighted metalicity
FF_ageLW={}
FF_ageMW={}
flux_dict={}
wave_dict={}
FF_wave={}
FF_flux={}

for pos, x in enumerate(id):
    sdss_file = '/home/zak/Documents/ProjectData/SDSS_Sample/spec-0266-51602-'+x+'.fits'

    hduspec = fits.open(sdss_file)
    t = hduspec['COADD'].data

    redshift = hduspec[2].data['Z']
    trialflux = t['flux']
    loglam_gal = t['loglam']
    galaxy = trialflux/np.median(trialflux)
    flux_dict[x] = galaxy

    trialwave = 10**loglam_gal
    restframe_wavelength = trialwave/(1+redshift)
    wave_dict[x] = restframe_wavelength
    vdisp = hduspec[2].data['vdisp']
    vdisp_round = int(np.round(vdisp/5.0)*5.0)# rounding vDisp for the models
    r_instrument = np.zeros(len(restframe_wavelength))
    invar =t['ivar'] # error for new training set


    if pos == 0:
        input_file='/home/zak/Documents/Project4Yr/MLstuff/MaStarTrainingTest.fits'
        hdu = fits.open(input_file)
        wavelength = hdu[1].data['wave']
        metal = hdu[1].data['Z']
        age = hdu[1].data['age']


    mask = np.where(np.logical_and(wavelength[0] < (np.max(restframe_wavelength)-2),wavelength[0] > (np.min(restframe_wavelength)+2))) #creating a mask to cut off the red end to match the observed data
    #-1 as the specres code seems to change slightly the last value of the input arrays causing a value error.
    newmodelwave = wavelength[0][mask]#universal wave array
    flux = hdu[1].data['features']
    r_model = (np.loadtxt(os.path.join(os.environ['FF_DIR'],'MaStar_SSP_v0.1_resolution_lin.txt')))[mask]
    wave_instrument = newmodelwave
    newinvar = sr.spectres(newmodelwave,restframe_wavelength,invar )#converting the invar array to the new wavelength positions
    standdev = np.sqrt(newinvar**-1)

    for wi,w in enumerate(newmodelwave):
    	r_instrument[wi] = 1900
    newmodelflux=[]

    for i in range(len(flux[:])): #cuting the red end off of the flux as well to match the upper wavelength limit of the observed

        fluxmask=flux[i]
        mf = downgrade_MaStar(newmodelwave,fluxmask[mask],r_model[:,1],vdisp_round, wave_instrument, r_instrument)


        newmodelflux.append(mf)

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

    #metal = hdu[1].data['Z']
    #age = hdu[1].data['age']

#--------------------------------------------
    #training the model
#--------------------------------------------
    xg_reg_metal = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 7, alpha=10,  n_estimators = 10)
    fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(newmodelflux,metal,random_state=1)
    #fluxTrain,fluxTest,metalTrain,metalTest = train_test_split(fluxext,metalext,random_state=1)
    xg_reg_metal.fit(fluxTrain,metalTrain)

    xg_reg_age = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 7, alpha=10,  n_estimators = 10)
    fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(newmodelflux,age,random_state=1)
    #fluxTrain,fluxTest,ageTrain,ageTest = train_test_split(fluxext,ageext,random_state=1)
    xg_reg_age.fit(fluxTrain,ageTrain)


#--------------------------------------------



#--------------------------------------------
    #removing the continuum of the observered
#--------------------------------------------
    interp = sp.interpolate.UnivariateSpline(restframe_wavelength, galaxy, s = 70)
    sub_flux = galaxy - interp(restframe_wavelength)
    #plt.plot(restframe_wavelength,sub_flux)
    #converting obsereved flux to the common wavelength newmodelwave
    newflux = sr.spectres(newmodelwave,restframe_wavelength,sub_flux)




#-------------------------------------------
   #making the prediction
#-------------------------------------------



    gal = newflux #creating a 2d array for testing purposes as xgb doesnt like 1d.
    stack =np.stack((newflux,gal))

    pred_metal=xg_reg_metal.predict(stack)[0]
    pred_age = xg_reg_age.predict(stack)[0]

    ML_metal[x] = pred_metal
    ML_age[x] = pred_age

#---------------------------------------------
#getting the FF information
#---------------------------------------------


    FF_file = '/home/zak/Documents/Project4Yr/MLstuff/FFsample/spFly-spec-0266-51602-'+x+'.fits'
    hduFF = fits.open(FF_file)

    FF_metalLW[x] = hduFF[1].header['metallicity_lightW']
    FF_metalMW[x] = hduFF[1].header['metallicity_massW']

    FF_ageLW[x] = hduFF[1].header['age_lightW']
    FF_ageMW[x] = hduFF[1].header['age_massW']

    FF_wave[x] = hduFF[1].data['wavelength']
    FF_flux[x] = hduFF[1].data['firefly_model']


#---------------------------------------------------------
# Comparison plots for metal and ages
#---------------------------------------------------------


# x = np.linspace(-2, 2, 100)
# y = x
# count=0
# plt.plot(x,y,'--',c='r',label='x = y')
# plt.scatter(list(FF_metalLW.values()),list(ML_metal.values()),marker='o',color='C0',label = 'FF_LightW')
# plt.scatter(list(FF_metalMW.values()),list(ML_metal.values()),marker='^',color='C1',label = 'FF_MassW')
# for p,q in zip(list(FF_metalLW.values()),list(ML_metal.values())):
#
#     label = list(ML_metal.keys())[count]
#
#     plt.annotate(label, # this is the text
#                  (p,q), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,5), # distance from text to points (x,y)
#                  ha='right') # horizontal alignment can be left, right or center
#     count = count+1
# plt.xlim(0.1,0.4)
# plt.ylim(-1.5,1.5)
# plt.xlabel('[Z/H] - FIREFLY')
# plt.ylabel('[Z/H] - ML Code')
# plt.legend()
# plt.show()
#
# x = np.linspace(-2, 2, 100)
# y = x
# count=0
# plt.plot(x,y,'--',c='r',label='x = y')
# plt.scatter(list(FF_ageLW.values()),list(ML_age.values()),marker='o',color='C0',label = 'FF_LightW')
# plt.scatter(list(FF_ageMW.values()),list(ML_age.values()),marker='^',color='C1',label = 'FF_MassW')
# for p,q in zip(list(FF_ageLW.values()),list(ML_age.values())):
#
#     label = list(ML_age.keys())[count]
#
#     plt.annotate(label, # this is the text
#                  (p,q), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,5), # distance from text to points (x,y)
#                  ha='right') # horizontal alignment can be left, right or center
#     count = count+1
# count = 0
# for p,q in zip(list(FF_ageMW.values()),list(ML_age.values())):
#
#     label = list(ML_age.keys())[count]
#
#     plt.annotate(label, # this is the text
#                  (p,q), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,5), # distance from text to points (x,y)
#                  ha='right') # horizontal alignment can be left, right or center
#     count = count+1
# plt.xlim(-2,2)
# plt.ylim(-2,9)
# plt.xlabel('[Z/H] - FIREFLY')
# plt.ylabel('[Z/H] - ML Code')
# plt.legend()
#plt.show()
ML_age
ML_metal
FF_metalLW
FF_metalMW
FF_ageLW
FF_ageMW
#-------------------------------------------------------------
# MaStar Plotter for given ML results
#-------------------------------------------------------------
fig, axs = plt.subplots(2,1, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.4, wspace=0.2)
axs = axs.ravel()
for pos, x in enumerate(id):

    ver='v0.1'
    lib_in='th'
    tin=float(ML_age[x])
    Zin=float(ML_metal[x])
    sin=float(1.8) #kr imf
    if (lib_in=='th' or lib_in=='Th' or lib_in=='TH' or lib_in=='theoretical'):
    	lib='Th'
    if (lib_in=='e' or lib_in=='E' or lib_in=='empirical'):
    	lib='E'

    t=MaStar_SSP.t(ver) #age array
    Z=MaStar_SSP.Z(ver) #metallicty array
    s=MaStar_SSP.s(ver) #IMF slope array
    wave=MaStar_SSP.wave(ver) #wavelength array
    fluxgrid=MaStar_SSP.flux(ver,lib) #flux matrix



    M_flux=MaStar_SSP.inter(t,Z,s,fluxgrid,tin,Zin,sin),
    norm_flux = M_flux/np.median(M_flux)
    FF_flux[x] = FF_flux[x]/np.median(FF_flux[x])

    interp1 = sp.interpolate.UnivariateSpline(wave_dict[x], flux_dict[x], s = 70)
    sub_flux1 = flux_dict[x] - interp1(wave_dict[x])
    interp2 = sp.interpolate.UnivariateSpline(FF_wave[x], FF_flux[x], s = 70)
    sub_flux2 = FF_flux[x] - interp2(FF_wave[x])
    interp3 = sp.interpolate.UnivariateSpline(wave, norm_flux, s = 70)
    sub_flux3 = norm_flux - interp3(wave)
    print(np.shape(wave))

    line_labels = ["ML MaStar Fit", "Input Spectra", "Firefly Fit"]
    l1 = axs[pos].plot(wave,sub_flux3[0],c='C2',label='ML MaStar Fit')
    l2 = axs[pos].plot(wave_dict[x],sub_flux1,c='C0',label='Input Spectra')
    l3 = axs[pos].plot(FF_wave[x],sub_flux2,c='C1',label='Firefly Fit')
    axs[pos].set_title('0266-51602-'+x)
    #fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.legend([l1,l2,l3], labels = line_labels, loc='center right')

    #axs[pos].set_xlabel('Wavelength (Angstrom)')
    #axs[pos].set_ylabel('Flux (1e-17erg/cm^2/s/Ang)')

    plt.show()
fig.add_subplot(111, frame_on=False)
#plt.legend([l1,l2,l3], labels = line_labels, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel("Flux (1e-17erg/cm^2/s/Ang)",labelpad=20)
plt.xlabel("Wavelength (Angstrom)")
plt.title('Continuum Removed', pad = 20, fontsize = 16)

plt.show()


#notes

# comparison with dust redding

#invar
#perturb flux using gaussian function based of the invar
#loop through flux array, take a gaussian function (look up python) width is invar
