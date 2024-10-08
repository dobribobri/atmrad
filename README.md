# atmrad
A framework for solving forward and inverse problems of atmospheric microwave radiometry 
based on International Telecommunication Union Recommendations 
<a href='https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-3-199708-S!!PDF-E.pdf'>ITU-R P.676-3</a>, 
<a href='https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-12-201908-S!!PDF-E.pdf'>ITU-R P.676-12</a>, 
<a href='https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-8-201908-I!!PDF-E.pdf'>ITU-R P.840-8</a>, etc. 

Atmrad includes two subpackages, which implement the same interface. 
CPU package uses only standard Python tools and numpy. GPU package additionally involves <a href="https://www.tensorflow.org/">TensorFlow</a> to operate with <a href="https://developer.nvidia.com/cuda-toolkit">NVIDIA CUDA</a>.

<pre>
See also <a href='https://github.com/dobribobri/meteo-'>https://github.com/dobribobri/meteo-</a>
</pre>

&nbsp;
#### Usage examples

<ul>
<li>Forward simulation of brightness temperatures in a sub-satellite point</li>
&nbsp;
  
Code:
  
```python
import numpy as np
from matplotlib import pyplot as plt

from cpu.atmosphere import Atmosphere
from cpu.surface import SmoothWaterSurface
import cpu.satellite as satellite
from cpu.cloudiness import CloudinessColumn


# A model of the Earth's atmosphere with standard altitude profiles of
# thermodynamic temperature, pressure and air humidity
# up to H = 20 km height with discretization of 500 nodes.
# Clear sky, liquid water content (LWC) is zero
atmosphere = Atmosphere.Standard(H=20., dh=20./500)
# The integration method and other parameters can be additionally specified
atmosphere.integration_method = 'boole'

# Introduce a smooth water surface as an underlying one
surface = SmoothWaterSurface()
# Specify surface temperature and salinity
surface.temperature = 15.
surface.salinity = 0.

# Set a frequency range e.g. from 10 to 150 GHz
frequencies = np.linspace(10, 150, 500)
# Compute "underlying surface - atmosphere" system outgoing radiation
# brightness temperatures at the specified frequencies
brt = satellite.brightness_temperatures(frequencies, atmosphere, surface, cosmic=True)

# Make a plot
plt.figure()
plt.xlabel(r'Frequencies $\nu$, GHz')
plt.ylabel(r'Brightness temperature $T_b$, K')
# Draw curve #1
plt.plot(frequencies, brt, ls='-', lw=2, label='Standard atmosphere, clear sky, LWC is zero')

# Consider a horizontally-homogeneous cloudiness of 1 km power with base altitude of 1.2 km.
# Find the corresponding liquid water altitude profile using Mazin's model.
# Liquid water amount is taken to be zero outside the cloud layer
liquid_water_distribution = CloudinessColumn(20., 500, clouds_bottom=1.2).liquid_water(height=1.)
# Assign it to the atmosphere object
atmosphere.liquid_water = liquid_water_distribution[0, 0, :]

# Repeat the brightness temperature calculation
brt = satellite.brightness_temperatures(frequencies, atmosphere, surface, cosmic=True)
# Draw curve #2
plt.plot(frequencies, brt, ls='--', lw=2,
         label='Cloud layer of 1 km power, LWC is {:.2f}'.format(np.round(atmosphere.W, decimals=2)) + ' kg/m$^2$')


# And one more time
liquid_water_distribution = CloudinessColumn(20., 500, clouds_bottom=1.2).liquid_water(height=3.)
atmosphere.liquid_water = liquid_water_distribution[0, 0, :]
brt = satellite.brightness_temperatures(frequencies, atmosphere, surface, cosmic=True)
# Draw curve #3
plt.plot(frequencies, brt, ls='-.', lw=2,
         label='Cloud layer of 3 km power, LWC is {:.2f}'.format(np.round(atmosphere.W, decimals=2)) + ' kg/m$^2$')

plt.grid(ls=':', alpha=0.5)
plt.legend(loc='best', frameon=False)
plt.tight_layout()
plt.savefig('example1.png', dpi=300)
plt.show()

```

&nbsp;

Result:

![example1](https://github.com/dobribobri/atmrad/assets/31748247/c01b57ee-8e0c-40af-bccf-05fbebe65d42)



<li>The inverse problem of total water vapor (TWV) and liquid water content (LWC) radiometric retrieval in a sub-satellite point. Common interface for this functionality is still under development.</li>
&nbsp;

Code:

```python

import numpy as np
from cpu.atmosphere import Atmosphere, avg
from cpu.cloudiness import CloudinessColumn
from cpu.surface import SmoothWaterSurface
import cpu.satellite as satellite
from cpu.weight_funcs import krho
from cpu.core.static.weight_funcs import kw


# Solve the forward problem firstly.
# See the previous example
atmosphere = Atmosphere.Standard(H=20., dh=20./500)
atmosphere.integration_method = 'boole'
liquid_water_distribution = CloudinessColumn(20., 500, clouds_bottom=1.2).liquid_water(height=1.)
atmosphere.liquid_water = liquid_water_distribution[0, 0, :]
surface = SmoothWaterSurface()

# We use further the dual-frequency method for TWV and LWC retrieval from the known brightness temperatures
frequency_pair = [22.2, 27.2]

# Obtain brightness temperatures for the specified frequency pair
brts = []
for nu in frequency_pair:
    brts.append(satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True))
brts = np.asarray(brts)


# Let's proceed to the inverse problem.
# Make some precomputes and model estimates
# Cosmic relict background
T_cosmic = 2.72548

sa = Atmosphere.Standard(H=20., dh=20./500)
# Water vapor specific attenuation coefficient (weighting function)
k_rho = [krho(sa, nu) for nu in frequency_pair]
# Liquid water specific attenuation coefficient (weighting function).
# Cloud effective temperature is taken to be 0 deg. Celsius here
k_w = [kw(nu, t=0.) for nu in frequency_pair]

M = np.asarray([k_rho, k_w]).T

# Total opacity coefficient in dry oxygen
tau_oxygen = np.asarray([sa.opacity.oxygen(nu) for nu in frequency_pair])
# Average "absolute" temperature of standard atmosphere downwelling radiation
T_avg_down = np.asarray([avg.downward.T(sa, nu) for nu in frequency_pair])
# Average "absolute" temperature of standard atmosphere upwelling radiation
T_avg_up = np.asarray([avg.upward.T(sa, nu) for nu in frequency_pair])
# Surface reflectivity index
R = np.asarray([surface.reflectivity(nu) for nu in frequency_pair])
# Surface emissivity under conditions of thermodynamic equilibrium
kappa = 1 - R

A = (T_avg_down - T_cosmic) * R
B = T_avg_up - T_avg_down * R - np.asarray(surface.temperature + 273.15) * kappa

D = B * B - 4 * A * (brts - T_avg_up)


# Compute total opacity from the known brightness temperatures and the model estimates
tau_experiment = -np.log((-B + np.sqrt(D)) / (2 * A))

# The dual-frequency method consists in resolving the following system of linear equations
sol = np.linalg.solve(M, tau_experiment - tau_oxygen)

# Display the retrieved total water vapor and liquid water content values
print('TWV is {:.2f} g/cm2, \t\t'.format(np.round(sol[0], decimals=2)) +
      'LWC is {:.2f} kg/m2'.format(np.round(sol[1], decimals=2)))

```

&nbsp;

Result:

<pre>
  TWV is 1.55 g/cm2, 		LWC is 0.12 kg/m2
</pre>


<li>Forward simulation of brightness temperature (map) outgoing from an atmspheric cell filled with cumuli</li>
&nbsp;

Code:
```python

import numpy as np
from matplotlib import pyplot as plt

# Using GPU acceleration
from gpu.atmosphere import Atmosphere
from gpu.surface import SmoothWaterSurface
import gpu.satellite as satellite
from cpu.cloudiness import Plank3D

frequency = 36      # GHz

# Standard atmosphere model with a smooth water surface as an underlying one
atmosphere = Atmosphere.Standard(H=20., dh=20./500)
atmosphere.integration_method = 'boole'
surface = SmoothWaterSurface()

# Let the atmospheric cell has sizes 50x50x20 km (Ox x Oy x Oz).
# We introduce a 3D computational grid of 300x300x500 nodes.
# Then fill the cell with cumuli distributed according to the "L2" case from (Plank, 1969), see tbl. 3
# For this case, the Plank model parameters are as follows:
alpha = 1.44    # a parameter depending on the time of day and various local climatic conditions (km^-1)
Dm = 4.026      # maximum effective cloud diameter in a population (km)
dm = 0.02286    # minimum effective cloud diameter (km)
eta = 0.93      # influences on cloud power (n.d.)
beta = 0.3      # also influences on cloud power (n.d.)
xi = -np.exp(-alpha * Dm) * (((alpha * Dm) ** 2) / 2 + alpha * Dm + 1) + \
    np.exp(-alpha * dm) * (((alpha * dm) ** 2) / 2 + alpha * dm + 1)
p = 0.65      # total sky cover ratio
K = 2 * np.power(alpha, 3) * (50 * 50 * p) / (np.pi * xi)     # effective number density
cloud_base = 1.2192      # cloud base altitude

# Generate the corresponding liquid water 3D-distribution
liquid_water_distribution = Plank3D(kilometers=(50., 50., 20.),
                                    nodes=(300, 300, 500),
                                    clouds_bottom=cloud_base).liquid_water(
    alpha=alpha, Dm=Dm, dm=dm, eta=eta, beta=beta, K=K,
)
atmosphere.liquid_water = liquid_water_distribution

# Obtain the brightness temperature map at the specified frequency
brt = satellite.brightness_temperature(frequency, atmosphere, surface, cosmic=True)

# Display it
plt.figure()
plt.imshow(brt.numpy())
plt.xlabel('nodes (Ox direction)')
plt.ylabel('nodes (Oy direction)')
plt.colorbar(label=r'$T_b$, K')
plt.tight_layout()
plt.savefig('example3.png', dpi=300)
plt.show()
```

&nbsp;

Result:

![example3](https://github.com/dobribobri/atmrad/assets/31748247/ebcd15ca-f87c-4d10-b860-5b6010cdd98c)



</ul>


#### References

1. B.G. Kutuza, M.V. Danilychev and O.I. Yakovlev, Satellite Monitoring of the Earth: Microwave Radiometry of Atmosphere and Surface [in Russian]. Moscow, Russia: Lenand Publ., 2016, 336 p.
2. V.G. Plank, The size distribution of cumulus clouds in representative Florida populations, J. Appl. Met., vol. 8, no. 1, pp. 46-67, 1969.
3. Reference standard atmospheres, document Recommendation ITU-R P.835-6, International Telecommunication Union, 2017.
4. Attenuation by atmospheric gases (Question ITU-R 201/3), document Recommendation ITU-R P.676-12, International Telecommunication Union, 2019.
5. Attenuation due to clouds and fog, document Recommendation ITU-R P.840, International Telecommunication Union, 2019.
6. D.H. Staelin, Measurements and interpretation of the microwave spectrum of the terrestrial atmosphere near 1‐centimeter wavelength, J. Geophys. Res., vol. 71, iss. 12, pp. 2875-2881, 1966.
7. Ed R. Westwater, The accuracy of water vapor and cloud liquid determination by dual-frequency ground-based microwave radiometry, Radio Science, vol. 13, no. 4, pp. 677-685, 1978.
8. D.P. Egorov and B.G. Kutuza, Atmospheric brightness temperature fluctuations in the resonance absorption band of water vapor 18-27.2 GHz, IEEE Trans. Geosci. Remote Sens., vol. 59, iss. 9, pp. 7627-7634, 2021.
9. B.G. Kutuza and M.T. Smirnov, The influence of clouds on radio-thermal radiance of atmosphere - ocean surface system, Issled. Zemli Kosm. [in Russian], no. 3, pp. 76-83, 1980.
10. S.P. Gagarin and B.G. Kutuza, Influence of sea roughness and atmospheric inhomogeneities on microwave radiation of the atmosphere -- ocean system, IEEE J. Ocean., vol. OE-8, no. 2, pp. 62-70, 1983.
11. I.P. Mazin and S.M. Shmeter, Clouds, Structure and Formation Physics [in Russian]. Leningrad, USSR: Gidromteoizdat, 1983, 279 p.
    
