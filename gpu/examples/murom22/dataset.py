
# -*- coding: utf-8 -*-
from cpu.core.static.water.vapor import absolute_humidity
from gpu.atmosphere import Atmosphere
from cpu.cloudiness import Cloudiness3D
from gpu.surface import SmoothWaterSurface
import gpu.satellite as satellite
import dill
import os
import numpy as np
from scipy import interpolate


with open('Dolgoprudnyj.dump', 'rb') as dump:
    radiosonde_data = dill.load(dump)

keys = list(radiosonde_data.keys())
grid = np.linspace(0.3, 14.7, 49)
T_tensor, P_tensor, rho_tensor = [], [], []
exceptions = []
for j, key in enumerate(keys):
    try:
        # По данным радиозондов
        T, P, rel, alt = radiosonde_data[key]
        max_km = 21
        T, P, rel, alt = T[alt < max_km], P[alt < max_km], rel[alt < max_km], alt[alt < max_km]
        rho = absolute_humidity(T, P, rel)
        T0 = T[0] + 6.5 * alt[0]
        P0 = P[0] / np.exp(-alt[0] / 7.7)
        rho0 = rho[0] / np.exp(-alt[0] / 2.1)
        f_T, f_P, f_rho = interpolate.interp1d(alt, T),  interpolate.interp1d(alt, P), interpolate.interp1d(alt, rho)
        T_grid, P_grid, rho_grid = f_T(grid), f_P(grid), f_rho(grid)
        T_grid = np.insert(T_grid, 0, T0)
        P_grid = np.insert(P_grid, 0, P0)
        rho_grid = np.insert(rho_grid, 0, rho0)
        # _grid = np.copy(grid)
        # _grid = np.insert(_grid, 0, 0.)
        T_grid, P_grid, rho_grid = \
            T_grid.astype(np.float32), P_grid.astype(np.float32), rho_grid.astype(np.float32)
        # _grid = _grid.astype(np.float32)
        T_tensor.append(T_grid)
        P_tensor.append(P_grid)
        rho_tensor.append(rho_grid)
        # alt_tensor.append(_grid)
        print('\r{:.2f}%'.format((j+1) / len(keys) * 100), end='   ', flush=True)

    except Exception as e:
        exceptions.append((j, e))

# print('\nExceptions at: {}'.format([j for j, _ in exceptions]))
print('Total: {}\t\t\tErrors: {}'.format(len(keys), len(exceptions)))

print('\n\nforming dataset...')
T_tensor, P_tensor, rho_tensor = \
    np.asarray([T_tensor], dtype=np.float32), np.asarray([P_tensor], dtype=np.float32), \
    np.asarray([rho_tensor], dtype=np.float32)
# alt_tensor = np.asarray([alt_tensor], dtype=np.float32)

print('T_tensor ', T_tensor.shape)
# print('alt_tensor ', alt_tensor.shape)

#########################################################################
# domain parameters
X = 50  # (км) горизонтальная протяженность моделируемой атмосферной ячейки

# observation parameters
angle = 0
incline = 'left'
integration_method = 'trapz'

# initial atmosphere parameters
T0 = 15.
P0 = 1013
rho0 = 7.5

# initial surface parameters
surface_temperature = 15.
surface_salinity = 0.

# radiation parameters
polarization = None
# frequencies = np.arange(1., 200., 0.2)
frequencies = [22.2, 27.2, 36, 89]

# liquid water distribution parameters
const_w = False
mu0 = 3.27
psi0 = 0.67

_c0 = 0.132574
_c1 = 2.30215

cl_bottom = 1.5

#########################################################################

__grid = np.linspace(0.3, 15., 50)
print('altitude grid ', __grid.shape)

atmosphere = Atmosphere(Temperature=T_tensor, Pressure=P_tensor, AbsoluteHumidity=rho_tensor,
                        altitudes=__grid)

atmosphere.integration_method = integration_method  # метод интегрирования

# atmosphere.angle = 30. * np.pi / 180.             # зенитный угол наблюдения, по умолчанию: 0
atmosphere.angle = angle
atmosphere.horizontal_extent = X
atmosphere.incline = incline  # в какую сторону по Ox наклонена траектория наблюдения

surface = SmoothWaterSurface(temperature=surface_temperature,
                             salinity=surface_salinity,
                             polarization=polarization)  # модель гладкой водной поверхности
surface.angle = atmosphere.angle

#########################################################################

nx, ny, nz = T_tensor.shape

W = np.arange(0., 5., 0.1)
H = np.power(W / _c0, 1. / _c1)

H_tensor = np.zeros((len(H), ny), dtype=np.float32)
for i, h in enumerate(H):
    H_tensor[i, :] = np.ones(ny) * h

lw_tensor = Cloudiness3D(kilometers=(X, X, 15), nodes=(len(H), ny, nz),
                         clouds_bottom=cl_bottom).liquid_water(
                             np.asarray(H_tensor),
                             const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
                        )
atmosphere.liquid_water = lw_tensor
Q_tensor = np.asarray([atmosphere.Q[0, :]] * len(H), dtype=np.float32)
print('Q ', Q_tensor.shape)
W_tensor = np.asarray(atmosphere.W, dtype=np.float32)
print('W ', W_tensor.shape)

# surfaceT_tensor = np.ones((len(H), ny), dtype=np.float32) * surface_temperature
surfaceT_tensor = np.asarray([T_tensor[0, :, 0]] * len(H), dtype=np.float32)
print('surfaceT_tensor ', surfaceT_tensor.shape)
surface.temperature = surfaceT_tensor

surfaceP_tensor = np.asarray([P_tensor[0, :, 0]] * len(H), dtype=np.float32)
surfaceRho_tensor = np.asarray([rho_tensor[0, :, 0]] * len(H), dtype=np.float32)
print('surfaceP_tensor ', surfaceP_tensor.shape)
print('surfaceRho_tensor ', surfaceRho_tensor.shape)

DBRT_tensor, OBRT_tensor = [], []
for j, nu in enumerate(frequencies):
    print('\r{:.2f}%'.format( j / (len(frequencies) - 1) * 100. ), end='   ', flush=True)
    d_brt = atmosphere.downward.brightness_temperature(nu, background=True)
    o_brt = satellite.brightness_temperature(nu, atmosphere, surface)
    DBRT_tensor.append(d_brt)
    OBRT_tensor.append(o_brt)
DBRT_tensor = np.asarray(DBRT_tensor, dtype=np.float32)
DBRT_tensor = np.moveaxis(DBRT_tensor, 0, -1)
OBRT_tensor = np.asarray(OBRT_tensor, dtype=np.float32)
OBRT_tensor = np.moveaxis(OBRT_tensor, 0, -1)
print('\nDBRT ', DBRT_tensor.shape)
print('OBRT ', OBRT_tensor.shape)

print('\nShaping dataset...')
with open('dataset.bin', 'wb') as dump:
    dill.dump((T_tensor, P_tensor, rho_tensor, __grid,
               surfaceT_tensor, surfaceP_tensor, surfaceRho_tensor,
               DBRT_tensor, OBRT_tensor, Q_tensor, W_tensor),
              dump, recurse=True)
