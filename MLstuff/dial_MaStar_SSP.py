
# Python routine that plots MaStar SSP spectrum for given version number, library, age, metallicity, and IMF slope
# Call with: python dial_MaStar_SSP.py <version number> <library> <age> <metallicity> <IMF slope>
# Example: python dial_MaStar_SSP.py v0.1 th 10.5 0.1 1.55
# Daniel Thomas, 14/11/2019

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import MaStar_SSP
import sys
ver=str(sys.argv[1])
lib_in=str(sys.argv[2])
tin=float(sys.argv[3])
Zin=float(sys.argv[4])
sin=float(sys.argv[5])

if (lib_in=='th' or lib_in=='Th' or lib_in=='TH' or lib_in=='theoretical'):
	lib='Th'
if (lib_in=='e' or lib_in=='E' or lib_in=='empirical'):
	lib='E'

t=MaStar_SSP.t(ver) #age array
Z=MaStar_SSP.Z(ver) #metallicty array
s=MaStar_SSP.s(ver) #IMF slope array
wave=MaStar_SSP.wave(ver) #wavelength array
fluxgrid=MaStar_SSP.flux(ver,lib) #flux matrix

print()
print('Plotting '+lib+'-MaStar SSP ('+ver+') spectrum with age='+sys.argv[2]+' Gyr, [Z/H]='+sys.argv[3]+' dex, IMF slope s='+sys.argv[4])
print()
print(len(wave))
print(wave)

flux=MaStar_SSP.inter(t,Z,s,fluxgrid,tin,Zin,sin)
plt.plot(wave,flux)
plt.xlabel('wavelength')
plt.ylabel('flux')
plt.show()
