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
         label='Cloud layer of 1 km power, LWC is {:.2f}'.format(atmosphere.W) + ' kg/m$^2$')


# And one more time
liquid_water_distribution = CloudinessColumn(20., 500, clouds_bottom=1.2).liquid_water(height=3.)
atmosphere.liquid_water = liquid_water_distribution[0, 0, :]
brt = satellite.brightness_temperatures(frequencies, atmosphere, surface, cosmic=True)
# Draw curve #3
plt.plot(frequencies, brt, ls='-.', lw=2,
         label='Cloud layer of 3 km power, LWC is {:.2f}'.format(atmosphere.W) + ' kg/m$^2$')

plt.grid(ls=':', alpha=0.5)
plt.legend(loc='best', frameon=False)
plt.savefig('example1.png', dpi=300)
plt.show()

```

&nbsp;

Result:

![example1](https://github.com/dobribobri/atmrad/assets/31748247/e851e87c-d18f-493e-b7fc-42568e7183a7)


<li>The inverse problem of total water vapor (TWV) and liquid water content (LWC) radiometric retrieval in a sub-satellite point</li>

Common interface for this functionality is still under development.
&nbsp;

Code:

```python

```

...

</ul>
