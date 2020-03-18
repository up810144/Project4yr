#Training data

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import scipy as sp
from scipy.signal import find_peaks
'''
modelfile = '/home/zak/Documents/ProjectData/MaStar_SSP_v0.1.fits'
hdu = fits.open(modelfile)
wavelength = hdu[2].data
print(wavelength)
print(max(wavelength))
t = hdu[1].data[:,0,0,0]
Z = hdu[1].data[0,:,0,1]
s = hdu[1].data[0,0,:,2]
fluxgrid = hdu[3].data
slope = [s[3]]
outputfile = 'MaStarTrainingTest.fits'

print('Preparing table...')
print(slope)
date=[]
for i in range(198):
    date.append('29-02-2020')



col0 = fits.Column(name='CreationDate', format='10A', array=date)
col1 = fits.Column(name='Age', format='1E')
col2 = fits.Column(name='Z', format='1E') #J = integer
col3 = fits.Column(name='slope', format='1E')#E = real set to kr imf slope
col4 = fits.Column(name='Wave', format='4563D',) #E = real set to kr imf slope
col5 = fits.Column(name='Flux', format='4563D')
col6 = fits.Column(name='Features', format='4563D')
coldef = fits.ColDefs([col0,col1,col2,col3,col4,col5,col6])
table = fits.BinTableHDU.from_columns(coldef)
k = 0
for i in range(len(t)):
    for j in range(len(Z)):
        wavelength
        age = t[1]
        metal = Z[2]
        flux_int = fluxgrid[1,2,3,:]
        print(flux_int)
        if max(flux_int) > -99:

            flux_normal = flux_int/np.median(flux_int)
            print(np.shape(flux_normal))

            interp = sp.interpolate.UnivariateSpline(wavelength,flux_normal, s = 20)
            sub_flux = flux_normal - interp(wavelength)


            table.data['Age'][k]=age
            table.data['slope'][k]=s[3]
            table.data['Z'][k]=metal
            table.data['Wave'][k]=wavelength
            table.data['Flux'][k]=flux_normal
            table.data['Features'][k]=sub_flux

            k=k+1


table.writeto(outputfile, overwrite=True)
'''
modelfile = '/home/zak/Documents/Project4Yr/MLstuff/NewMaStarModels.fits'
hdu = fits.open(modelfile)
wave = hdu[1].data['wave']
wavelength = wave[1]
t = hdu[1].data['Age']
Z = hdu[1].data['Z']
s = hdu[1].data['slope']
flux=hdu[1].data['Flux']
slope = hdu[1].data['slope']
outputfile = 'MaStarTrainingTestv2.fits'

print('Preparing table...')
date=[]
for i in range(16067):
    date.append('16-03-2020')


col0 = fits.Column(name='CreationDate', format='10A', array=date)
col1 = fits.Column(name='Age', format='1E')
col2 = fits.Column(name='Z', format='1E') #J = integer
col3 = fits.Column(name='slope', format='1E')#E = real set to kr imf slope
col4 = fits.Column(name='Wave', format='4563D',) #E = real set to kr imf slope
col5 = fits.Column(name='Flux', format='4563D')
col6 = fits.Column(name='Features', format='4563D')
coldef = fits.ColDefs([col0,col1,col2,col3,col4,col5,col6])
table = fits.BinTableHDU.from_columns(coldef)
k = 0
for i in range(len(t)):

    age = t[i]
    metal = Z[i]
    flux_int = flux[i]


    if max(flux_int) > -99:

        flux_normal = flux_int/np.median(flux_int)

        interp = sp.interpolate.UnivariateSpline(wavelength,flux_normal, s = 70)
        sub_flux = flux_normal - interp(wavelength)

        table.data['Age'][k]=age
        table.data['slope'][k]=slope[1]
        table.data['Z'][k]=metal
        table.data['Wave'][k]=wavelength
        table.data['Flux'][k]=flux_normal
        table.data['Features'][k]=sub_flux

        k=k+1


table.writeto(outputfile, overwrite=True)
