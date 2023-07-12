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
<li>Forward simulation of brightness temperature in a sub-satellite point</li>
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
# taken up to H = 20 km height with discretization of 100 nodes.
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
plt.savefig('example1.png', dpi=300)
plt.show()

```

&nbsp;

Result:

![example1](https://github.com/dobribobri/atmrad/assets/31748247/e851e87c-d18f-493e-b7fc-42568e7183a7)


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

# We use further dual-frequency method for TWV and LWC retrieval from the known brightness temperatures
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
# Liquid water specific attenuation coefficient (weighting function)
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

</ul>
