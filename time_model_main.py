"""
Created on Sun Mar  4 17:32:57 2018
Main script for evalating time dependent model at VO's
@author: eyu
"""
import numpy as np
from lib import GMT_tools as gmt
import pandas as pd
from lib import functions as ft
import matplotlib.pyplot as plt
from lib import functions as ft

# ------------------------------------------------- LOADING -------------------------------------------------------

# Data paths, loaded in next loop.
# read list of tau(n)

file = '/Users/eyu/Google Drive/DTU/6_semester/Bachelor/Bachelorproject/data/tau_spect_av_GUFM-SAT-E3.txt'
tau = pd.read_table(file, header=None, delim_whitespace=True)

# naming columns
tau.columns = ['degree', 'tau_MF [years]', 'tan_SV ([years]']

tau = tau['tau_MF [years]'].values * 1

file1 = '/Users/eyu/Google Drive/DTU/6_semester/Bachelor/Bachelorproject/data/VO_CHAMP_MF0101.txt'
file2 = '/Users/eyu/Google Drive/DTU/6_semester/Bachelor/Bachelorproject/data/VO_SWARM_MF_0101.txt'
files = [file1, file2]

# ------------------------------------------------- SET VALUES -------------------------------------------------------
degree = 13
N = degree * (degree + 2)
r_surface = 6371.2
r_core = 3480.
deg = 180/np.pi

# times of data
tp = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2014, 2015, 2016, 2017]
epochs = [3, 7, 11]

# reference times
tq = np.arange(start=2004, stop=2019)

# loads the r-component of the field, and indicdes for the chosen VO's
[theta, phi, indices] = ft.get_vo_indices(files=files, year=2017, epoch=3, vo_coordinates=[(48.17388, 8.7721),
                                                                                           (143.524, 86.00912),
                                                                                           (95.96354000000001, 123.77911)])

# creates a list identical design matrices, and quantifies the times of data (epochs per year):
[Gs, Grs_cmb, tps, Bs] = ft.design_SHA_per_epoch(data_paths=files, year_list=tp, epochs=epochs, degree=degree)

# placeholder for later plotting
tps_data = tps

# creat spatio-temporal design matrix
A = ft.design_SHA_GP(Gi_t=Gs, prediction_times=tps, reference_times=tq, tau=tau, degree=degree)

# solve inverse problem
a = A.T.dot(A)
b = A.T.dot(Bs)
model = np.linalg.solve(a, b)

B_synth = A.dot(model)
residuals = Bs - B_synth
ft.errors_plot(residuals)

# ------------------------------------------------- MODEL EVALUATION --------------------------------------------------
# creates a linearly space grid in time, for which the model predict the field:

time_resolution = 4
time_grid = np.arange(start=2006, stop=2019, step=1/time_resolution)
time_span = int((len(time_grid)/time_resolution))

print('creating Gr synth')
Gs = ft.design_time_grid(data_paths=files, year_list=tp, epochs=epochs, degree=degree, grid=time_grid)

print('creating A synth')
A = ft.design_SHA_GP(Gi_t=Gs, prediction_times=time_grid, reference_times=tq, tau=tau, degree=degree)

print('creating synthetic field')
Br_synth = A.dot(model)
print('synthetic field finished')


# ------------------------------------------------- PLOTTING -------------------------------------------------------
# The plotting routine used is iterating through subplots
# ------------------------------------------------- FIELD -------------------------------------------------------
# Plots the field component over the three chosen VO's given by the variable indices
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=1, wspace=0.3)
time_mask = np.where(np.logical_or(time_grid < 2011, time_grid > 2014))

# ranges matches the data at VO
list_of_ranges = [[-34000, -30500], [42500, 48000], [16000, 18500]]
axs = axs.ravel()

for i in range(len(indices)):
    Br_synVO = Br_synth[indices[i][0]::894]
    Br_data = Bs[indices[i][0]::894]
    axs[i].plot(time_grid[time_mask], Br_synVO[time_mask], 'r.', label='VO prediction')
    axs[i].plot(tps_data, Br_data, 'b+', label='VO data')
    # axs[i].set_ylim(list_of_ranges[i])
    axs[i].set_xticks([2005, 2008, 2011, 2014, 2017])
    axs[i].grid(b=True, which='major', color='k', linestyle='-')
    axs[i].grid(b=True, which='minor', linestyle='--')
    axs[i].minorticks_on()
    axs[i].legend()

    axs[i].set_xlabel('Years')
    ylabel_string = r'$\bf{B_r}$' + ', [nT]'
    axs[i].set_ylabel(ylabel_string)

    string = 'VO, ' + r"$(\theta, \varphi) = $" + '(' + str(np.round(theta[indices[i][0]] * deg, 3)) + ', ' \
             + str(np.round(phi[indices[i][0]] * deg, 3)) + ')'
    axs[i].set_title(string, fontsize=12)

# plt.savefig('Br_three_VOs.png')
plt.show()

# ------------------------------------------------- SV -------------------------------------------------------
# Plots the secular variation over the three chosen VO's given by the variable indices
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=1, wspace=0.3)

# ranges matches the data at VO
list_of_ranges = [[-75, 50], [-50, 75], [25, -100]]

axs = axs.ravel()
for i in range(len(indices)):
    # loads a single VO
    [Br, Bt, Bp, times, sv, sv_time] = ft.load_single_vo(files=files, Bi='Br',
                                                         theta=theta[indices[i][0]] * deg, phi=phi[indices[i][0]] * deg)

    Br_synth_tmp = Br_synth[indices[i][0]::894]

    sv_synth = np.empty([time_span - 1, 0])
    sv_synth_time = np.empty([time_span - 1, 0])

    # crates a matrix with size (year x time resolution)
    for t in range(time_resolution):
        sv_synth_tmp = np.diff(Br_synth_tmp[t::time_resolution])
        sv_synth = np.hstack((sv_synth, sv_synth_tmp.reshape(-1, 1)))

        current_time = time_grid[t::time_resolution]
        current_time = (current_time[:-1] + current_time[1::]) / 2
        sv_synth_time = np.hstack((sv_synth_time, current_time.reshape(-1, 1)))

    # flattens the array to get a timely chronological order on the secular variation.
    sv_synth = sv_synth.flatten()
    sv_synth_time = sv_synth_time.flatten()

    time_mask = np.where(np.logical_or(sv_synth_time < 2010, sv_synth_time > 2014))

    axs[i].plot(sv_time, sv, 'b+', label='VO data')
    axs[i].plot(sv_synth_time[time_mask], sv_synth[time_mask], 'r.', label='GP model')
    axs[i].set_ylim(list_of_ranges[i])
    axs[i].set_xticks([2005, 2008, 2011, 2014, 2017])
    axs[i].grid(b=True, which='major', linestyle='-')
    axs[i].grid(b=True, which='minor', linestyle='--')
    axs[i].legend()

    axs[i].set_xlabel('Years')
    ylabel_string = r'$\bf{B_r}$' + ', [nT]/yr'
    axs[i].set_ylabel(ylabel_string)

    string = 'VO, ' + r"$(\theta, \varphi) = $" + '(' + str(np.round(theta[indices[i][0]] * deg, 3)) + ', ' \
             + str(np.round(phi[indices[i][0]] * deg, 3)) + ')'
    axs[i].set_title(string, fontsize=12, fontweight='bold')

# plt.savefig('SV_three_VOs.png')
plt.show()

