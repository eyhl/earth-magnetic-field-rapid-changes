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


#
# def design_SHA_GP(Gi_t, prediction_times, reference_times, tau, degree):
#     ''' This function computes the spatial temporal design matrix.
#     Inputs:
#     Gi_t                =  A list of spatial design matrices
#     prediction_times    =  list of holding either times of data, or times of wished for predictions
#     reference_times     =  list of reference times
#     tau                 =  characteristic time scale
#     degree              =  spherical harmonic degree
#
#     output: A
#     '''
#     # For handling tau as integer input:
#     try:
#         len_tau = len(tau)
#     except TypeError:
#         tau = [tau]
#
#     n_coefficients = degree * (degree + 2)
#     n_ref_times = len(reference_times)
#
#     # Creates the k* matrix as defined in Rasmussen and Williamson eq. 2.25 combined with spherical harmonic design
#     # matrix to obtain spatial-temporal design matrix called A.
#     test = 1
#     A = np.empty([0, n_coefficients * n_ref_times])
#     for tp_idx, tp_val in enumerate(prediction_times):
#         # for holding the complete k* matrix at a given tp
#         cov_matrix_tp = np.empty([np.size(Gi_t[tp_idx], axis=0), 0])
#
#         for coeff_count, gr_col in enumerate(Gi_t[tp_idx].T):
#             # for temporarily storing the current coefficient times tq.
#             ref_time_columns = np.empty([np.size(Gi_t[tp_idx], axis=0), 0])
#
#             # Determines based on the current order, if the degree has increased:
#             current_degree = -1 + np.sqrt(1 + coeff_count)
#             if (current_degree - int(current_degree)) == 0:
#                 if len(tau) == 1:
#                     tau_idx = 0
#                 else:
#                     tau_idx = int(current_degree)
#
#             # creates a new column for every kernel(tp, tq_i) * current_coefficient_data_column.
#             for tq_idx, tq_val in enumerate(reference_times):
#                 ref_time_columns = np.hstack((ref_time_columns, (gr_col * ft.matern_kernel_element(tp_val, tq_val, [tau[tau_idx]])).reshape(-1, 1)))
#                 if test == 1:
#                     # print(tp_val, tq_val)
#                     # print((gr_col * ft.matern_kernel_element(tp_val, tq_val, [tau[tau_idx]])).reshape(-1, 1))
#                     test -= 1
#
#             cov_matrix_tp = np.hstack((cov_matrix_tp, ref_time_columns))
#
#         # stacks for each tp.
#         A = np.vstack((A, cov_matrix_tp))
#
#     return A
#
#
# def design_SHA_per_epoch(data_paths, year_list, epochs, degree, truncation=None):
#     ''' This function loads data, and creates a list of design matrices, field values and quantified times.
#     An element in the list corresponds to a time.
#
#     Inputs:
#     data_paths          =  list of paths to data
#     year_list           =  list years in question
#     epochs              =  value for given epochs
#     degree              =  spherical harmonic degree
#     truncation          =  mainly for testing, enables truncating the data input
#
#     output: Grs, tps, Brs, indices
#     '''
#
#     # HYPER-PARAMETERS
#     r_surface = 6371.2  # earths mean radius, often called a
#     epoch_as_decimal = (1 - (12 - 12/len(epochs)) / 12)
#
#
#     # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
#     # division by zero are ignored.
#     Grs = []
#     Brs = np.empty([0])
#     tps = []
#     for y in year_list:
#         for e in epochs:
#             [Br, theta, phi, r, indices] = ft.load_epochs_r(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
#                                                             truncation=truncation)
#             if len(Br) == 300:
#                 # Only Gr is used so division by zero is ignored.
#                 with np.errstate(divide='ignore', invalid='ignore'):
#                     [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)
#
#                 # create list of Gr design matrices, one for each tp
#                 Grs.append(Gr)
#
#                 # stack Brs for residuals
#                 Brs = np.hstack((Brs, Br))
#                 # if len(Br) == 300:
#                 # print(y, e)
#                 # quantifying tp and append to list.
#                 if e == 3:
#                     tps.append(y + epoch_as_decimal)
#                 elif e == 7:
#                     tps.append(y + 2 * epoch_as_decimal)
#                 else:
#                     tps.append(y + 2.99999999 * epoch_as_decimal)
#
#     return Grs, np.array(tps), Brs, indices
#
#
# def design_time_grid(data_paths, year_list, epochs, degree, grid, truncation=None):
#     ''' This function loads data, and creates a list of design matrices, based on a full data set.
#
#     Inputs:
#     data_paths          =  list of paths to data
#     year_list           =  list years in question
#     epochs              =  value for given epochs
#     degree              =  spherical harmonic degree
#     grid                =  the number of times that the design matrix should be repeated.
#     truncation          =  mainly for testing, enables truncating the data input
#
#     output: Grs, tps, Brs, indices
#     '''
#
#     # HYPER-PARAMETERS
#     r_surface = 6371.2  # earths mean radius, often called a
#     epoch_as_decimal = (1 - (12 - 12/len(epochs)) / 12)
#
#
#     # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
#     # division by zero are ignored.
#     Grs = []
#     Brs = np.empty([0])
#     tps = []
#     for y in year_list:
#         for e in epochs:
#             [Br, theta, phi, r, indices] = ft.load_epochs_r(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
#                                                             truncation=truncation)
#             if len(theta) == 300:
#                 if y > 2013:
#                     # Only Gr is used so division by zero is ignored.
#                     with np.errstate(divide='ignore', invalid='ignore'):
#                         [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)
#
#                     # create list of Gr design matrices, one for each tp
#                     Gr_swarm_template = Gr
#                 else:
#                     [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)
#                     Gr_champ_template = Gr
#
#     # print(np.shape(Gr_champ_template))
#     # print(np.shape(Gr_swarm_template))
#     for t in grid:
#         if t <= 2014:
#             Grs.append(Gr_champ_template)
#         else:
#             Grs.append(Gr_swarm_template)
#
#     return Grs

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
[Br, theta, phi, r, indices] = ft.load_epochs_r(files=files, year=2017, epoch=3)

# creates a list identical design matrices, and quantifies the times of data (epochs per year):
[Grs, tps, Brs, indices] = ft.design_SHA_per_epoch(data_paths=files, year_list=tp, epochs=epochs, degree=degree)

# placeholder for later plotting
tps_data = tps

# creat spatio-temporal design matrix
A = ft.design_SHA_GP(Gi_t=Grs, prediction_times=tps, reference_times=tq, tau=tau, degree=degree)

# solve inverse problem
a = A.T.dot(A)
b = A.T.dot(Brs)
model = np.linalg.solve(a, b)


# ------------------------------------------------- MODEL EVALUATION --------------------------------------------------
# creates a linearly space grid in time, for which the model predict the field:

time_resolution = 4
time_grid = np.arange(start=2006, stop=2019, step=1/time_resolution)
time_span = int((len(time_grid)/time_resolution))

print('creating Gr synth')
Grs = ft.design_time_grid(data_paths=files, year_list=tp, epochs=epochs, degree=degree, grid=time_grid)

print('creating A synth')
A = ft.design_SHA_GP(Gi_t=Grs, prediction_times=time_grid, reference_times=tq, tau=tau, degree=degree)

print('creating synthetic field')
Br_synth = A.dot(model)
print('synthetic field finished')


# ------------------------------------------------- PLOTTING -------------------------------------------------------
# The plotting routine used is iterating through subplots
# ------------------------------------------------- FIELD -------------------------------------------------------
# Plots the field component over the three chosen VO's given by the variable indices
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=1, wspace=0.3)

# ranges matches the data at VO
list_of_ranges = [[-34000, -30500], [42500, 48000], [16000, 18500]]
axs = axs.ravel()

for i in range(len(indices)):
    Br_synVO = Br_synth[indices[i][0]::300]
    Br_data = Brs[indices[i][0]::300]
    axs[i].plot(time_grid, Br_synVO, 'r--', label='VO prediction')
    axs[i].plot(tps_data, Br_data, 'b.', label='VO data')
    axs[i].set_ylim(list_of_ranges[i])
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

plt.savefig('Br_three_VOs.png')
plt.show()

# ------------------------------------------------- SV -------------------------------------------------------
# Plots the secular variation over the three chosen VO's given by the variable indices
fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=1, wspace=0.3)

# ranges matches the data at VO
list_of_ranges = [[-100, 100], [-50, 150], [80, -160]]

axs = axs.ravel()
for i in range(len(indices)):
    # loads a single VO
    [Br, Bt, Bp, times, sv, sv_time] = ft.load_single_VO(files=files, Bi='Br',
                                                         theta=theta[indices[i][0]] * deg, phi=phi[indices[i][0]] * deg)

    Br_synth_tmp = Br_synth[indices[i][0]::300]

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

    axs[i].plot(sv_time, sv, 'b.', label='VO data')
    axs[i].plot(sv_synth_time, sv_synth, 'r--', label='GP model')
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

plt.savefig('SV_three_VOs.png')
plt.show()

