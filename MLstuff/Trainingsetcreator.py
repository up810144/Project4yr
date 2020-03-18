#MaStar ssp enhancer

# Python routine that plots MaStar SSP spectrum for given version number, library, age, metallicity, and IMF slope
# Call with: python dial_MaStar_SSP.py <version number> <library> <age> <metallicity> <IMF slope>
# Example: python dial_MaStar_SSP.py v0.1 th 10.5 0.1 1.55
# Daniel Thomas, 14/11/2019

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import MaStar_SSP
import sys
ver='v0.1'
lib_in='th'
tin=float(0.1) #Min age
Zin=float(-2.35) #min Z
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
print(len(wave))

outputfile ='NewMaStarTrainingset.fits'

date=[]

for p in range(16445):
    date.append('16-03-2020')

col0 = fits.Column(name='CreationDate', format='10A', array=date)
col1 = fits.Column(name='Age', format='1E')
col2 = fits.Column(name='Z', format='1E') #J = integer
col3 = fits.Column(name='slope', format='1E')#E = real set to kr imf slope
col4 = fits.Column(name='Wave', format='4563D',) #E = real set to kr imf slope
col5 = fits.Column(name='Flux', format='4563D')

coldef = fits.ColDefs([col0,col1,col2,col3,col4,col5])
table = fits.BinTableHDU.from_columns(coldef)

k=0
for i in range(299):
    Zin = -2.35
    for j in range(55):

        age = tin
        metal = Zin
        slope = sin
        flux=MaStar_SSP.inter(t,Z,s,fluxgrid,tin,Zin,sin)

        table.data['Age'][k]=age
        table.data['slope'][k]=slope
        table.data['Z'][k]=metal
        table.data['Wave'][k]=wave
        table.data['Flux'][k]=flux


        k=k+1
        Zin = Zin + 0.05

    tin = tin + 0.05

table.writeto(outputfile, overwrite=True)
