"""
Created on Sun Mar  4 17:32:57 2018
The following toolbox contains the functions used for spherical harmonic analysis for geomagnetism.
@author: eyu
"""
import numpy as np
from lib import GMT_tools as gmt
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
# -------------------------------------------------- SPATIAL TOOLS -------------------------------------------------

def load_single_epoch(files, year, epoch, header_size=4, NaN_marker=99999.):
    '''
    This functions loads in data from the global magnetic field for a chosen year and epoch (3,7,11).
    Data points should have following units:
    theta, phi - in degrees
    r          - in km
    Br, Bt, Bp - in nT

    The delimeter is defaulted to be whitespaces and the structure of the input file is assumed to be:
    [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    Values at the poles (theta = 0 and 180) are assumed erroneous and their rows are removed.
    NaN marker is set to default as 99999, and all rows including NaNs are removed.
    note: theta and phi is returned in radians, no other unit conversion is done.

    :return: Br, Bt, Bp, theta, phi, r
    '''
    # for conversion into radians
    rad = np.pi / 180

    # for selecting chosen year and epoch
    year = 'Year==' + str(year)
    epoch = 'Month==' + str(epoch)

    # reads data
    list_of_dataframes = []
    for file in files:
        # read file
        df = pd.read_table(file, skiprows=header_size, delim_whitespace=True)

        # naming columns
        df.columns = ['theta', 'phi', 'Year', 'Month', 'Time', 'r', 'Br', 'Bt', 'Bp', 'N_{data}']

        list_of_dataframes.append(df)

    # concatenate dataframes into one
    data = pd.concat(list_of_dataframes, axis=0, ignore_index=True)

    # selects chosen data time (year, epoch)
    data_time = data.query(year)
    data_time = data_time.query(epoch)

    # removes pole data points, and NaNs
    # data_time = data_time[data_time['theta'] != 0]  # drop rows with theta=0
    # data_time = data_time[data_time['theta'] != 180]  # drop rows with theta=180
    data_time = data_time.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data_time = data_time.dropna(how='any')  # drop rows with NaNs

    # save grid variables based on data points
    Br = data_time['Br'].values
    Bt = data_time['Bt'].values
    Bp = data_time['Bp'].values
    theta = data_time['theta'].values * rad  # convert to radians
    phi = data_time['phi'].values * rad
    r = data_time['r'].values

    return Br, Bt, Bp, theta, phi, r


def load_single_VO(files, Bi, theta, phi, header_size=4, NaN_marker=99999.):
    '''
    This functions loads in data from the global magnetic field for a chosen year and epoch (3,7,11).
    Data points should have following units:
    theta, phi - in degrees
    r          - in km
    Br, Bt, Bp - in nT

    The delimeter is defaulted to be whitespaces and the structure of the input file is assumed to be:
    [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    Values at the poles (theta = 0 and 180) are assumed erroneous and their rows are removed.
    NaN marker is set to default as 99999, and all rows including NaNs are removed.
    note: theta and phi is returned in radians, no other unit conversion is done.

    :return: Br, Bt, Bp, theta, phi, r
    '''
    # for conversion
    rad = np.pi / 180
    one_third_year = (1 - (12 - 4) / 12)

    # rounds input to three decimal to ensure that the coordinates are found in data.
    theta = np.round(theta, 3)
    phi = np.round(phi, 3)

    # reads data
    list_of_dataframes = []
    for file in files:
        # read file
        df = pd.read_table(file, skiprows=header_size, delim_whitespace=True)

        # naming columns
        df.columns = ['theta', 'phi', 'Year', 'Month', 'Time', 'r', 'Br', 'Bt', 'Bp', 'N_{data}']

        list_of_dataframes.append(df)

    # concatenate dataframes into one
    data = pd.concat(list_of_dataframes, axis=0, ignore_index=True)

    # round theta and phi values for selection of coordinates
    data = data.round({'theta': 3, 'phi': 3})

    # selects chosen VO coordinates (theta, phi)
    thetas = data['theta'] == theta
    phis = data['phi'] == phi
    data_time = data[thetas & phis]

    # handles NaNs
    data_time = data_time.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data_time.loc[data_time.isnull().any(axis=1), 0:2] = np.nan
    data_time.loc[data_time.isnull().any(axis=1), 3::] = np.nan
    data_time = data_time.fillna(value=0)  # fill NaNs with zeros

    # every year appears several times in file
    years = np.unique(data_time['Year'].values)

    sv_time = []
    sv = []
    # This loop takes epoch for year y and check if the same epoch for year y + 1 exists. If so it is used for sv.
    for y in years:
        df1 = data_time.query('Year==' + str(y))
        df2 = data_time.query('Year==' + str(y + 1))
        months1 = df1['Month'].values
        months2 = df2['Month'].values
        for idx, m in enumerate(months1):
            # try to look at epoch for next year, if year not existant it will continue
            try:
                if m != 0:
                    if m == months2[idx]:
                        if m == 3:
                            sv_time.append(((y + one_third_year) + (y + 1 + one_third_year)) / 2)
                            sv1 = (df1.loc[df1['Month'] == m, Bi]).values
                            sv2 = (df2.loc[df2['Month'] == m, Bi]).values
                            sv.append(sv2[0] - sv1[0])
                        elif m == 7:
                            sv_time.append(((y + 2*one_third_year) + (y + 1 + 2*one_third_year)) / 2)
                            sv1 = (df1.loc[df1['Month'] == m, Bi]).values
                            sv2 = (df2.loc[df2['Month'] == m, Bi]).values
                            sv.append(sv2[0] - sv1[0])
                        elif m == 11:
                            sv_time.append(((y + 3*one_third_year) + (y + 1 + 3*one_third_year)) / 2)
                            sv1 = (df1.loc[df1['Month'] == m, Bi]).values
                            sv2 = (df2.loc[df2['Month'] == m, Bi]).values
                            sv.append(sv2[0] - sv1[0])
            except IndexError:
                pass

    data_time = data_time.replace(0, np.nan)  # set all 99999 values to NaN
    data_time = data_time.dropna(how='any')

    column_name = 'Month'
    mask_epoch3 = data_time.Month == 3
    mask_epoch7 = data_time.Month == 7
    mask_epoch11 = data_time.Month == 11
    data_time.loc[mask_epoch3, column_name] = one_third_year
    data_time.loc[mask_epoch7, column_name] = 2 * one_third_year
    data_time.loc[mask_epoch11, column_name] = 2.9999999 * one_third_year
    #
    # count_full_years = data_time.groupby('Year').count()
    # data_time = data_time.loc[data_time['Year'].isin(count_full_years[count_full_years['Month'] == 3].index)]

    times = (data_time['Year'] + data_time['Month']).values

    # save data in seperate variables
    Br = data_time['Br'].values
    Bt = data_time['Bt'].values
    Bp = data_time['Bp'].values
    r = data_time['r'].values

    return Br, Bt, Bp, times, sv, sv_time


def load_epochs_r(files, year, epoch, truncation=None, header_size=4, NaN_marker=99999.):
    '''
    This functions loads in data from the global magnetic field for a chosen year and epoch (3,7,11), and returns
    lists containing Br component and corresponding theta, phi and r.
    Data points should have following units:
    The input columns should have units: theta and phi in degrees, r in km and Br in nT

    Input parameters:
    files       - list of data files
    year        - chosen year of data, this year has to exist in one of the files
    epoch       - chosen epoch of data, this epoch has to exist in one of the files
    truncation  - set to a number in order to truncate the data, e.g. 1 in order to only look at the first data point.
    header_size - number of header rows in the data files
    NaN_marker  - change if data contains a different NaN marker than 99999.

    The delimeters of the files are defaulted to be whitespaces and the structure of the input file is assumed to be:
    [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    NaN marker is set to default as 99999, and all rows including NaNs are removed.
    note: theta and phi is returned in radians.
    :return: Br, theta, phi, r
    '''
    # TODO udvid med alle komponenter, drop nan og 0 og 180 lat på baggrund af if else.

    # conversion into radians
    rad = np.pi / 180

    # selecting chosen year and epoch
    y = year
    year = 'Year==' + str(year)
    epoch = 'Month==' + str(epoch)

    # reads data
    list_of_dataframes = []
    for file in files:
        # read file
        df = pd.read_table(file, skiprows=header_size, delim_whitespace=True)

        # naming columns
        df.columns = ['theta', 'phi', 'Year', 'Month', 'Time', 'r', 'Br', 'Bt', 'Bp', 'N_{data}']

        list_of_dataframes.append(df)

    # concatenate dataframes into one
    data = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
    data.loc[data['phi'] < 0, 'phi'] += 360 # note added +360, yields more stable B

    single_year_mask = data['Year'] == 2017
    single_epoch_mask = data['Month'] == 3
    single_year = data[single_year_mask & single_epoch_mask].reset_index()
    g = single_year.groupby(['theta', 'phi'])
    index1 = g.get_group((48.17388, 8.7721)).index.values
    index2 = g.get_group((143.524, 86.00912)).index.values
    index3 = g.get_group((95.96354000000001, 123.77911)).index.values
    indices = [index1, index2, index3]

    # selects chosen data time (year, epoch)
    data_time = data.query(year)
    data_time = data_time.query(epoch)
    # removes pole data points, and NaNs
    data_time = data_time.drop(['Bt', 'Bp', 'N_{data}'], axis=1)
    data_time = data_time.reset_index()

    # data_time = data_time[data_time['theta'] != 0]  # drop rows with theta=0
    # data_time = data_time[data_time['theta'] != 180]  # drop rows with theta=180
    data_time = data_time.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data_time = data_time.dropna(how='any')  # drop rows with NaNs

    # save grid variables based on data points
    Br = data_time['Br'].values
    theta = data_time['theta'].values * rad # convert to radians
    phi = data_time['phi'].values * rad
    r = data_time['r'].values

    if truncation:
        k = truncation
        [Br, theta, phi, r] = [Br[0:k], theta[0:k], phi[0:k], r[0:k]]

    # if y == 2017:
    #     print(Br[index1], '\n', data_time.iloc[index1])
    #     print(Br[index2], '\n', data_time.iloc[index2])
    #     print(Br[index3], '\n', data_time.iloc[index3])

    return Br, theta, phi, r, indices


# def load_VO_coordinates(files, year, epoch, truncation=None, header_size=4, NaN_marker=99999.):
#     '''
#     This functions loads in data from the global magnetic field for a chosen year and epoch (3,7,11), and returns
#     lists containing Br component and corresponding theta, phi and r.
#     Data points should have following units:
#     The input columns should have units: theta and phi in degrees, r in km and Br in nT
#
#     Input parameters:
#     files       - list of data files
#     year        - chosen year of data, this year has to exist in one of the files
#     epoch       - chosen epoch of data, this epoch has to exist in one of the files
#     truncation  - set to a number in order to truncate the data, e.g. 1 in order to only look at the first data point.
#     header_size - number of header rows in the data files
#     NaN_marker  - change if data contains a different NaN marker than 99999.
#
#     The delimeters of the files are defaulted to be whitespaces and the structure of the input file is assumed to be:
#     [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
#     NaN marker is set to default as 99999, and all rows including NaNs are removed.
#     note: theta and phi is returned in radians.
#     :return: Br, theta, phi, r
#     '''
#     # conversion into radians
#     rad = np.pi / 180
#
#     # selecting chosen year and epoch
#     year = 'Year==' + str(year)
#     epoch = 'Month==' + str(epoch)
#
#     # reads data
#     list_of_dataframes = []
#     for file in files:
#         # read file
#         df = pd.read_table(file, skiprows=header_size, delim_whitespace=True)
#
#         # naming columns
#         df.columns = ['theta', 'phi', 'Year', 'Month', 'Time', 'r', 'Br', 'Bt', 'Bp', 'N_{data}']
#
#         list_of_dataframes.append(df)
#
#     # concatenate dataframes into one
#     data = pd.concat(list_of_dataframes, axis=0, ignore_index=True)
#
#     # selects chosen data time (year, epoch)
#     data_time = data.query(year)
#     data_time = data_time.query(epoch)
#
#     # save grid variables based on data points
#     Br = data_time['Br'].values
#     Bt = data_time['Bt'].values
#     Bp = data_time['Bp'].values
#     theta = data_time['theta'].values * rad # convert to radians
#     phi = data_time['phi'].values * rad
#     r = data_time['r'].values
#
#     if truncation:
#         k = truncation
#         [Br, Bt, Bp, theta, phi, r] = [Br[0:k], Bt[0:k], Bp[0:k], theta[0:k], phi[0:k], r[0:k]]
#
#
#     return Br, Bt, Bp, theta, phi, r


def global_field_model(Bi, Gi, L, degree, regularise='', alpha=1e-8, gamma=1, eps=1e-4):
    '''
    Computes global model of the geomagnetic field at the surface, given by least squares estimate:
        model = inverse(G.T.dot(G)).G.T.dot(d)
    In the case of regularisation the model will be given by:
        L1: model = inverse(G.T.dot(G) + alpha**2 * L.T.dot(Wm).dot(L)) .dot (G.T.dot(d)), where Wm = diag(1/gamma)
        L2: model = inverse(G.T.dot(G) + alpha**2 * L.T.dot(L)) .dot (G.T.dot(d))
    where L = Gr_cmb (core-mantle boundary), Wm = diag(1/gamma)

    Input parameters:
    Bi          - magnetic field component
    Gi          - corresponding design matrix component
    L           - regularisation term parameter, in geomagnetism often L = Gi_cmb
    degree      - the harmonic degree, N

    optional:
    regularise  - '' default - no regularisation. Set to 'L1' for L1 norm regularisation, set to 'L2' for L2 norm
                  regularisation.
    alpha       - regularisation parameter
    gamma       - the initial weights for Wm. Default 1
    eps         - the epsilon of Ekblom's measure, to avoid division by zero. Default 1e-4

    note: r, theta, phi has to be equal length

    :return: model, residuals, misfit_norm, model_norm
    '''
    N = len(Bi)

    # figure out what which dimension of L that is not the "number of model coefficients"
    size_array = np.array([np.size(L, 0), np.size(L, 1)])  # array with number of rows and columns in L
    correct_size = int(size_array[np.argwhere(size_array != degree*(degree + 2))])  # size != number of coefficients

    # defining the weight matrix for L1
    wm = np.ones(correct_size) * 1/np.sqrt(gamma ** 2 + eps ** 2)  # Ekblom's measure, avoid division by zero

    # right hand side (rhs) are similar in all cases
    rhs = Gi.T.dot(Bi)

    if regularise == 'L1':
        lhs = (Gi.T.dot(Gi) + alpha**2 * L.T.dot(np.diag(wm)).dot(L))  # left hand side (lhs)
        model = np.linalg.solve(lhs, rhs)

        # L1-norm defined as sum(abs(gamma[i])):
        model_norm = 1/N * np.sum(np.abs(gamma))

    elif regularise == 'L2':
        lhs = (Gi.T.dot(Gi) + alpha**2 * L.T.dot(L))  # left hand side (lhs)
        model = np.linalg.solve(lhs, rhs)

        # L2-norm defined as sum(L.dot(m)**2):
        model_norm = 1/N * np.sum(L.dot(model) ** 2)

    else:
        lhs = Gi.T.dot(Gi)  # left hand side (lhs)
        model = np.linalg.solve(lhs, rhs)
        model_norm = 0

    residuals = (Bi - Gi.dot(model))
    misfit_norm = 1/N * np.sqrt(residuals.T.dot(residuals))

    return model, residuals, misfit_norm, model_norm


def L1_norm(Bi, Gi, L, degree, alpha_list, gamma=1, eps=1e-4, converged=0.001, printall=False):
    '''
    This function computes L1-regularised models corresponding to a list of regularisation parameters (alphas). In this
    regard a weighting matrix is introduced.
    Introducing a weighting matrix, Wm, the optimum model coefficients of L1 norm regularisation is given by:
        model = (G.T.dot(G) - alpha**2 * L.T.dot(Wm).dot(L))**(-1).dot(G.T.dot(data))
    Where Wm is a diagonal matrix, Wm[i, i] = 1/gamma[i]
    The gammas are determined through the Iteratively Reweighted Least Squares (IRLS) method.
    The function returns lists for analysis and further processing of the all the models. To avoid computing the sought
    for model again afterwards, the list of models is outputted, and the sought for model will be placed at
    model_list[j].
    Note: that the model and misfit norms are normalised with the number of data points, len(Bi).

    Input parameters:
    B_i         - magnetic field component [nT]
    G_i         - design matrix component
    L           - linear model parameter, in geomagnetism: L = Gr_cmb (cmb; core mantle boundary).
    degree      - the harmonic degree, N
    alpha_list  - list of alphas to be iterated through

    optional:
    gamma       - the initial weights for Wm. Default 1
    eps         - the epsilon of Ekblom's measure, to avoid division by zero. Default 1e-4
    converged   - convergence limit. Default 0.01%
    printall    - If true convergence status of every iteration is printed, if set to 'no' nothin is printed.
                  Default: prints current alpha and every tenth convergence status.

    :return: model_list, residuals_list, misfit_list, model_norm_list, gamma_list
    '''
    # for normalisation of model and misfit norm
    N = len(Bi)


    gamma_list = []
    model_list = []
    residuals_list = []
    misfit_list = []
    model_norm_list = []

    # figure out what which dimension of L that is not the "number of model coefficients"
    size_array = np.array([np.size(L, 0), np.size(L, 1)])  # array with number of rows and columns in L
    correct_size = int(size_array[np.argwhere(size_array != degree*(degree + 2))])  # size != number of coefficients

    for alpha in alpha_list:
        if printall == 'n':
            pass
        else:
            print('current alpha: ', alpha)

        # for every new alpha, start up with initial guess:
        gamma = 1

        # initialise L1-regularised model in order to compute convergence condition in 1st while-loop:
        [model_previous, dummy, dummy, dummy] = global_field_model(Bi=Bi, Gi=Gi, L=L, degree=degree,
                                                                   regularise='L1', alpha=alpha, gamma=gamma)
        # set convergence arbitrarily higher than 0.001
        convergence = 1

        # counter for print option
        i = 0
        while convergence > converged:
            gamma = L.dot(model_previous)

            # defining the weight matrix for L1
            wm = np.ones(correct_size) * 1 / np.sqrt(gamma ** 2 + eps ** 2)

            # compute the model at current iteration
            rhs = Gi.T.dot(Bi)
            lhs = (Gi.T.dot(Gi) + alpha**2 * L.T.dot(np.diag(wm)).dot(L))
            model_current = np.linalg.solve(lhs, rhs)

            # compute convergence parameter
            convergence = np.sqrt(np.linalg.norm(model_current - model_previous) / np.linalg.norm(model_current))
            model_previous = model_current

            # prints every tenth or every single iteration
            i += 1
            if printall:
                print('convergence', convergence)
            elif printall == 'n':
                pass
            elif i % 10 == 1:
                print('convergence', convergence)

        # computes model norm, residuals and misfit for evaluation purposes
        model_norm = 1/N * np.sum(np.abs(gamma))
        residuals = (Bi - Gi.dot(model_current))
        misfit = 1/N * np.sqrt(residuals.T.dot(residuals))

        # list for finding the best model wrt given alpha
        gamma_list.append(gamma)
        model_list.append(model_current)
        residuals_list.append(residuals)
        misfit_list.append(misfit)
        model_norm_list.append(model_norm)

    return model_list, residuals_list, misfit_list, model_norm_list, gamma_list


def L2_norm(Bi, Gi, L, alpha_list):
    '''
    This function computes L2-regularised models corresponding to a list of regularisation parameters (alphas)
    For L2 norm regularisation of a model the optimum model coefficients can be given by:
        model = inverse(G.T.dot(G) + alpha**2 * L.T.dot(L)) .dot (G.T.dot(d))
    This function determines the weights in the the diagonal of Wm, through the IRLS method. To avoid computing the
    sougth for model again afterwards, the list of models is outputted, and the sought for model will be placed at
    model_list[j].
    Returns lists for analysis and further processing of the all the models.
    Note: that the model and misfit norms are normalised with the number of data points, len(Bi).
    B_i         - magnetic field component [nT]
    G_i         - design matrix component
    L           - linear model parameter, in geomagnetism: L = Gr_cmb.
                  (not to be confused with the L in L_curve)
    alpha_list  - list of alphas to be used

    :return: model_list, residuals_list, misfit_list, model_norm_list
    '''
    # for normalisation of model and misfit norm
    N = len(Bi)

    model_list = []
    misfit_list = []  # sometimes denoted rho
    model_norm_list = []  # sometimes denoted eta
    residuals_list = []

    for alpha in alpha_list:
        rhs = Gi.T.dot(Bi)
        lhs = Gi.T.dot(Gi) + alpha**2 * L.T.dot(L)
        model = np.linalg.solve(lhs, rhs)

        residuals = (Bi - Gi.dot(model))

        misfit = 1/N * np.sqrt(residuals.T.dot(residuals))

        model_norm = 1/N * np.sqrt(np.sum(L.dot(model) ** 2))

        misfit_list.append(misfit)
        model_norm_list.append(model_norm)
        residuals_list.append(residuals)
        model_list.append(model)

    return model_list, residuals_list, misfit_list, model_norm_list


def compute_G_cmb(stepsize, degree):
    '''
    Computes the G design matrix at the core mantle boundary to be used
    in the regularisation term of regularised least squares.
    stepsize    - step sized of the regular grid used for input in the design_SHA()
    degree      - harmonic degree
    :return: Gr_cmb, Gt_cmb, Gp_cmb
    '''

    r_surface = 6371.2  # earths mean radius, often called a.
    r_core = 3480.

    # computes a regular grid
    theta_cmb = np.arange(180 - stepsize / 2, 0, - stepsize)
    phi_cmb = np.arange(-180 + stepsize / 2, 180 - stepsize / 2, stepsize)
    theta_cmb_grid, phi_cmb_grid = np.meshgrid(theta_cmb, phi_cmb)

    theta_cmb_grid = np.reshape(theta_cmb_grid, np.size(theta_cmb_grid), 1)
    phi_cmb_grid = np.reshape(phi_cmb_grid, np.size(phi_cmb_grid), 1)
    length = len(theta_cmb_grid)

    # comutes design matrix at core mantle boundary (cmb), based on grid.
    [Gr_cmb, Gt_cmb, Gp_cmb] = gmt.design_SHA(r_core / r_surface * np.ones(length),
                                              theta_cmb_grid, phi_cmb_grid, degree)
    return Gr_cmb, Gt_cmb, Gp_cmb


def global_field_plot(model, model_name='not-specified', radius=6371.2, save=False, vmin=-1e6, vmax=1e6,
                      component='r', step=.25, projection_type='hammer', cmap='PuOr_r', lat0=0, lon0=0):
    '''
    This function plot the magnetic field on a global map.
    Input: model

    Options:
        - change a to look at model at different radii, default is the surface
          of the earth
        - to save plot, change: save = 'y'
        - change model_name to the model name inputtet for title on plot
        - field component: choose either 'r'(default), 'theta' or 'phi'/'theta'
        - step: stepsize of grid
        - projection_type: choose a projection type: 'hammer'(default),
          'robin', 'mill' etc.
        - cmap: choose colormap
        - vmin, vmax: min and max values of colorbar.
        - lat0 and lon0 specifies center of map.

    :return basemap plot of model with default setting
    '''
    # defines class in order to center zero at "central color" of the colorbar
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # Ignoring masked values and all kinds of edge cases
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    r = radius / 6371.2
    rad = np.pi / 180
    theta_bound = [-90, 90]
    phi_bound = [-180, 180]

    theta = theta_bound[1] - np.arange(theta_bound[0], theta_bound[1] + step, step)
    theta[np.where(theta == 0)] = 1e-15  # in order to avoid division by zero
    theta[np.where(theta == 180)] = 179.9999999999999999999  # avoiding division by zero in G_lambda
    phi = np.arange(phi_bound[0], phi_bound[1] + step, step)

    # grid of magnetic field at r=a (surface) up to spherical harmonic degree N
    B_r, B_theta, B_phi = gmt.synth_grid(model, r, theta * rad, phi * rad)

    if component == 'r':
        B_i = B_r
    elif component == 'theta':
        B_i = B_theta
    elif component == 'phi' or component == 'lambda':
        B_i = B_phi
    else:
        print('Component not recognized, choose r, theta or phi/lambda')

    # basemap plotting, check basemap documentation,
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    m = Basemap(projection=projection_type, resolution='l', area_thresh=1000.0,
                lat_0=lat0, lon_0=lon0)
    # m = Basemap(projection='spstere', boundinglat=-10, lon_0=0, resolution='l')
    # m = Basemap(projection='ortho', resolution='l', area_thresh=1000.0, lat_0=-90, lon_0=0)

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    xx, yy = m(phi_grid, theta_bound[1] - theta_grid)
    m.pcolormesh(xx, yy, B_i, cmap=cmap, norm=MidpointNormalize(midpoint=0.), vmin=vmin, vmax=vmax)
    m.drawcoastlines(linewidth=0.25)
    m.drawparallels(np.arange(-80, 81, 20), labels=[1, 0, 0, 0])
    # m.drawmeridians(np.arange(0, 360, 60), labels=[0, 0, 0, 1])  # not used for Hammer projection

    cbar = m.colorbar()
    cbar.set_label('B_r [nT]', rotation=0)

    title_string = ('Global magnetic field' + '\n' + model_name + '-model' + ', ' +
                    component + '-' + 'component ' + 'at ' +
                    str(radius) + ' km (from earth center)')
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    plt.title(title_string, fontsize=14, fontweight='bold')

    # optionally saves figure
    if save:
        string = ('global_field_' + model_name + '.png')
        plt.savefig(string)

    plt.show()


def errors_plot(residuals, choice=[True, False], latitude=False, convert=True, savefig=False):
    '''
    Plots errors of a model. Default plots only histogram.

    Input parameters:
    residuals   - residuals/errors of the model to be evaluated
    choice      - set to True according to the outputted plots
    latitude    - input latitude if choice[1] is True
    convert     - set to False if latitude is already in degrees

    :return: plot(s)
    '''
    if convert:
        try:
            theta = latitude * 180 / np.pi
        except NameError:
            print('Please input latitudes')

    if choice[0]:
        # best fit of data
        (mu, sigma) = norm.fit(residuals)

        # the histogram of the data
        n, bins, patches = plt.hist(residuals, bins=50, normed=1, facecolor='green')

        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'b--', linewidth=2)
        plt.title(r'$\bf{\mu=%.3f,\ \sigma=%.3f}$' % (mu, sigma), fontsize=14)
        plt.xlabel('Error values, [nT]', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        ax = plt.gca()
        ax.yaxis.grid(which='major', color='grey', linestyle='dashed')
        ax.xaxis.grid(which='major', color='grey', linestyle='dashed')
        ax.set_axisbelow(True)

        if savefig:
            plt.savefig('Error_histogram.png')

        plt.show()


    if choice[1]:
        plt.figure()
        plt.plot(residuals, theta, 'b.')
        plt.title('Residuals vs. co-latitude', fontsize=14, fontweight='bold')
        plt.xlabel('Residuals, [nT]', fontsize=14)
        plt.ylabel('Co-latitude, [degrees]', fontsize=14)
        plt.grid(True)

        if savefig:
            plt.savefig('Erros_vs_latitude.png')

        plt.show()

    return


def L_curve_plot(misfit_list, model_norm_list, alpha_index, point=True):
    '''
    Plots the L curve, and places a red dot on the chosen alpha as default

    Input parameters:
    misfit_list      - list of misfit values for models evaluated at different alphas
    model_norm_list  - list of model norm values for models evaluated at different alphas
    alpha_index      - the index of the alpha corresponding to a set of [misfit, model_norm]-coordinates.
    point            - set to False if the coordinate-point, corresponding to the chosen alpha should not be plotted.

    :return: plot
    '''

    fig, ax = plt.subplots()
    ax.plot(misfit_list, model_norm_list, 'b.', label='L-curve')
    # ax.loglog(misfit_norms_trunc, model_norms_trunc, 'g.', label='L-curve, truncated')
    if point:
        ax.plot(misfit_list[alpha_index], model_norm_list[alpha_index], 'ro', label='alpha^2')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Misfit 2-norm, log-scale')
    ax.set_ylabel('Model norm, log-scale')
    #
    # locmaj = ticker.LogLocator(base=10, numticks=12)
    # ax.xaxis.set_major_locator(locmaj)
    #
    # locmin = ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    # ax.xaxis.set_minor_locator(locmin)
    # ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ax.grid(b='on', which='major', color='k', linewidth=.5)
    # ax.grid(b='on', which='minor', color='k', linewidth=.25)
    ax.grid(True, which="both", ls="-")
    ax.minorticks_on()

    ax.set_title('L-curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()

    return


def L_curve_corner(rho, eta, alpha):
    '''
    Parameter Estimation and Inverse Problems, 2nd edition, 2011
    originally for Matlab by R. Aster, B. Borchers, C. Thurber
    converted to python by E. Lippert

    The functions is related to the regularised least squares problem, and the regularisation parameter.
    This function returns the optimum regularisation parameter value, ie the optimum alpha,
    by a maximum curvature (kappa) estimation.

    rho     - misfit term
    eta     - regularisation term/function
    alpha   - regularisation parameter

    :return scorner_alpha, corner_index, kappa
    '''

    # L-curve is defined in log-log space
    x = np.log(rho)
    y = np.log(eta)

    # if a input is list, it still works
    alpha = np.array(alpha)

    # Circumscribed circle simple approximation to curvature (after Roger Stafford)

    # vectors containing three sets of points from the input
    x1 = x[0:-2]
    x2 = x[1:-1]
    x3 = x[2::]
    y1 = y[0:-2]
    y2 = y[1:-1]
    y3 = y[2::]

    # the length of the sides of the triangles in the circumscribed circle
    a = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
    c = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    s = (a + b + c) / 2  # semi - perimeter

    # the radius of the circle
    R = (a * b * c) / (4 * np.sqrt(s * (s - a) * (s - b) * (s - c)))

    # the reciprocal of the radius yields the curvature for each estimate for each value
    kappa = np.pad(1 / R, (1, 1), 'constant')  # zero-padded: end-points has no curvature
    print(np.shape(kappa))
    print(np.argmax(kappa))
    print(kappa)

    corner_index = np.argmax(kappa)
    corner_alpha = alpha[corner_index]  # the optimum alpha as found in the L curve corner

    return corner_alpha, corner_index, kappa


def L_curve_geometrical_corner(rhos, etas):
    '''
    This function is not used at the moment.
    By 'drawing' a line between the first and the last point, it sets the corner
    to be the data point furthest away from this line.
    '''
    nPoints = len(rhos)
    allCoord = np.vstack((etas, rhos)).T
    np.array([etas, rhos])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)

    return idxOfBestPoint

# -------------------------------------------------- TIME DEPENDENCY -------------------------------------------------

def time_dep_synth_grid(model, r, theta, phi, degree, tps, tqs, tau):
    """
    Corresponds to the spatial synth grid function, but this takes a temporal design matrix into account.
    note: can only handle one prediction time (tps)
    """
    n_coeff = degree * (degree + 2)
    nmax = int(np.sqrt(n_coeff + 1) - 1)
    N_theta = len(theta)
    N_phi = len(phi)
    n = np.arange(0, nmax + 1)

    r_n = r ** (-(n + 2))

    cos_sin_m = np.ones((n_coeff, N_phi))
    sin_cos_m = np.zeros((n_coeff, N_phi))
    T_r = np.zeros((n_coeff, N_theta))
    T_theta = np.zeros((n_coeff, N_theta))
    T_phi = np.zeros((n_coeff, N_theta))

    Pnm = gmt.get_Pnm(nmax, theta)
    sinth = Pnm[1][1]
    sinth[np.where(sinth == 0)] = 1e-10  # in order to avoid division by zero

    k = 0
    for n in np.arange(1, nmax + 1):
        T_r[k] = (n + 1.) * r_n[n] * Pnm[n][0]
        T_theta[k] = -r_n[n] * Pnm[0][n + 1]
        k = k + 1
        for m in np.arange(1, n + 1):
            T_r[k] = (n + 1) * r_n[n] * Pnm[n][m]
            T_theta[k] = -r_n[n] * Pnm[m][n + 1]
            T_phi[k] = m * r_n[n] * Pnm[n][m] / sinth
            cos_sin_m[k] = np.cos(m * phi)
            sin_cos_m[k + 1] = cos_sin_m[k]
            T_r[k + 1] = T_r[k]
            T_theta[k + 1] = T_theta[k]
            T_phi[k + 1] = -T_phi[k]
            cos_sin_m[k + 1] = np.sin(m * phi)
            sin_cos_m[k] = cos_sin_m[k + 1]
            k = k + 2

    n_test_times = len(tps)
    n_ref_times = len(tqs)

    # Creates the k* matrix as defined in Rasmussen and Williamson eq. 2.25
    k_star = np.empty([0, n_coeff * n_ref_times])
    for tp_idx, tp_val in enumerate(tps):
        cov_matrix_tp = np.empty([np.size(T_r, axis=1), 0])
        for coefficient_count, gr_col in enumerate(T_r):
            ref_time_columns = np.empty([np.size(T_r, axis=1), 0])

            current_degree = -1 + np.sqrt(1 + coefficient_count)
            if (current_degree - int(current_degree)) == 0:
                tau_idx = int(current_degree)
                print(tau_idx)

            for tq_idx, tq_val in enumerate(tqs):
                ref_time_columns = np.hstack(
                    (ref_time_columns,
                     (gr_col * matern_kernel_element(tp_val, tq_val, [tau[tau_idx]])).reshape(-1, 1)))
            ref_time_columns = np.array(ref_time_columns)
            cov_matrix_tp = np.hstack((cov_matrix_tp, ref_time_columns))
        k_star = np.vstack((k_star, cov_matrix_tp))

    print(np.shape(k_star))
    repeat_list = list((np.ones(len(cos_sin_m)) * int(len(tqs))))
    cos_sin_m = np.repeat(cos_sin_m, repeat_list, axis=0)
    tmp = cos_sin_m * model[:, np.newaxis]
    B_r = np.matmul(k_star, tmp)

    # B_theta = np.matmul(T_theta.transpose(), tmp)
    #
    # repeat_list = list((np.ones(len(sin_cos_m)) * int(len(tqs))))
    # sin_cos_m = np.repeat(sin_cos_m, repeat_list, axis=0)
    # tmp = sin_cos_m * model[:, np.newaxis]
    # B_phi = np.matmul(T_phi.transpose(), tmp)
    B_theta = 1
    B_phi = 1

    return (B_r, B_theta, B_phi)


def time_dep_global_field_plot(model, test_times, reference_times, tau, model_name='not-specified', radius=6371.2,
                               save=False, vmin=-1e6, vmax=1e6, component='r', step=.25, projection_type='hammer',
                               cmap='PuOr_r', lat0=0, lon0=0):
    '''
    This function plot the magnetic field on a global map.
    Input: model

    Options:
        - change a to look at model at different radii, default is the surface
          of the earth
        - to save plot, change: save = 'y'
        - change model_name to the model name inputtet for title on plot
        - field component: choose either 'r'(default), 'theta' or 'phi'/'theta'
        - step: stepsize of grid
        - projection_type: choose a projection type: 'hammer'(default),
          'robin', 'mill' etc.
        - cmap: choose colormap
        - vmin, vmax: min and max values of colorbar.
        - lat0 and lon0 specifies center of map.

    :return basemap plot of model with default setting
    '''
    # defines class in order to center zero at "central color" of the colorbar
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # Ignoring masked values and all kinds of edge cases
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    r = radius / 6371.2
    rad = np.pi / 180
    theta_bound = [-90, 90]
    phi_bound = [-180, 180]
    degree = 13

    theta = theta_bound[1] - np.arange(theta_bound[0], theta_bound[1] + step, step)
    theta[np.where(theta == 0)] = 1e-15  # in order to avoid division by zero
    theta[np.where(theta == 180)] = 179.9999999999999999999  # avoiding division by zero in G_lambda
    phi = np.arange(phi_bound[0], phi_bound[1] + step, step)

    # grid of magnetic field at r=a (surface) up to spherical harmonic degree N
    B_r, B_theta, B_phi = time_dep_synth_grid(model=model, r=r, theta=theta * rad, phi=phi * rad, degree=degree,
                                              tps=test_times, tqs=reference_times, tau=tau)

    if component == 'r':
        B_i = B_r
    elif component == 'theta':
        B_i = B_theta
    elif component == 'phi' or component == 'lambda':
        B_i = B_phi
    else:
        print('Component not recognized, choose r, theta or phi/lambda')

    # basemap plotting, check basemap documentation,
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    m = Basemap(projection=projection_type, resolution='l', area_thresh=1000.0,
                lat_0=lat0, lon_0=lon0)
    # m = Basemap(projection='spstere', boundinglat=-10, lon_0=0, resolution='l')
    # m = Basemap(projection='ortho', resolution='l', area_thresh=1000.0, lat_0=-90, lon_0=0)

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    xx, yy = m(phi_grid, theta_bound[1] - theta_grid)
    m.pcolormesh(xx, yy, B_i, cmap=cmap, norm=MidpointNormalize(midpoint=0.), vmin=vmin, vmax=vmax)
    m.drawcoastlines(linewidth=0.25)
    m.drawparallels(np.arange(-80, 81, 20), labels=[1, 0, 0, 0])
    # m.drawmeridians(np.arange(0, 360, 60), labels=[0, 0, 0, 1])  # not used for Hammer projection

    cbar = m.colorbar()
    cbar.set_label('B_r [nT]', rotation=0)

    title_string = ('Global magnetic field' + '\n' + model_name + '-model' + ', ' +
                    component + '-' + 'component ' + 'at ' +
                    str(radius) + ' km (from earth center)')
    ttl = ax.title
    ttl.set_position([.5, 1.05])

    plt.title(title_string, fontsize=14, fontweight='bold')

    # optionally saves figure
    if save:
        string = ('global_field_' + model_name + '.png')
        plt.savefig(string)

    plt.show()


def design_SHA_GP(Gi_t, prediction_times, reference_times, tau, degree):
    ''' This function computes the spatial temporal design matrix.
    Inputs:
    Gi_t                =  A list of spatial design matrices
    prediction_times    =  list of holding either times of data, or times of wished for predictions
    reference_times     =  list of reference times
    tau                 =  characteristic time scale
    degree              =  spherical harmonic degree

    output: A
    '''
    # For handling tau as integer input:
    try:
        len_tau = len(tau)
    except TypeError:
        tau = [tau]

    n_coefficients = degree * (degree + 2)
    n_ref_times = len(reference_times)

    # Creates the k* matrix as defined in Rasmussen and Williamson eq. 2.25 combined with spherical harmonic design
    # matrix to obtain spatial-temporal design matrix called A.
    test = 1
    A = np.empty([0, n_coefficients * n_ref_times])
    for tp_idx, tp_val in enumerate(prediction_times):
        # for holding the complete k* matrix at a given tp
        cov_matrix_tp = np.empty([np.size(Gi_t[tp_idx], axis=0), 0])

        for coeff_count, gr_col in enumerate(Gi_t[tp_idx].T):
            # for temporarily storing the current coefficient times tq.
            ref_time_columns = np.empty([np.size(Gi_t[tp_idx], axis=0), 0])

            # Determines based on the current order, if the degree has increased:
            current_degree = -1 + np.sqrt(1 + coeff_count)
            if (current_degree - int(current_degree)) == 0:
                if len(tau) == 1:
                    tau_idx = 0
                else:
                    tau_idx = int(current_degree)

            # creates a new column for every kernel(tp, tq_i) * current_coefficient_data_column.
            for tq_idx, tq_val in enumerate(reference_times):
                ref_time_columns = np.hstack((ref_time_columns, (
                            gr_col * matern_kernel_element(tp_val, tq_val, [tau[tau_idx]])).reshape(-1, 1)))
                if test == 1:
                    # print(tp_val, tq_val)
                    # print((gr_col * ft.matern_kernel_element(tp_val, tq_val, [tau[tau_idx]])).reshape(-1, 1))
                    test -= 1

            cov_matrix_tp = np.hstack((cov_matrix_tp, ref_time_columns))

        # stacks for each tp.
        A = np.vstack((A, cov_matrix_tp))

    return A


def design_SHA_per_epoch(data_paths, year_list, epochs, degree, truncation=None):
    ''' This function loads data, and creates a list of design matrices, field values and quantified times.
    An element in the list corresponds to a time.

    Inputs:
    data_paths          =  list of paths to data
    year_list           =  list years in question
    epochs              =  value for given epochs
    degree              =  spherical harmonic degree
    truncation          =  mainly for testing, enables truncating the data input

    output: Grs, tps, Brs, indices
    '''

    # HYPER-PARAMETERS
    r_surface = 6371.2  # earths mean radius, often called a
    epoch_as_decimal = (1 - (12 - 12 / len(epochs)) / 12)

    # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
    # division by zero are ignored.
    Grs = []
    Brs = np.empty([0])
    tps = []
    for y in year_list:
        for e in epochs:
            [Br, theta, phi, r, indices] = load_epochs_r(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
                                                            truncation=truncation)
            if len(Br) == 300:
                # Only Gr is used so division by zero is ignored.
                with np.errstate(divide='ignore', invalid='ignore'):
                    [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

                # create list of Gr design matrices, one for each tp
                Grs.append(Gr)

                # stack Brs for residuals
                Brs = np.hstack((Brs, Br))
                # if len(Br) == 300:
                # print(y, e)
                # quantifying tp and append to list.
                if e == 3:
                    tps.append(y + epoch_as_decimal)
                elif e == 7:
                    tps.append(y + 2 * epoch_as_decimal)
                else:
                    tps.append(y + 2.99999999 * epoch_as_decimal)

    return Grs, np.array(tps), Brs, indices


def design_time_grid(data_paths, year_list, epochs, degree, grid, truncation=None):
    ''' This function loads data, and creates a list of design matrices, based on a full data set.

    Inputs:
    data_paths          =  list of paths to data
    year_list           =  list years in question
    epochs              =  value for given epochs
    degree              =  spherical harmonic degree
    grid                =  the number of times that the design matrix should be repeated.
    truncation          =  mainly for testing, enables truncating the data input

    output: Grs, tps, Brs, indices
    '''

    # HYPER-PARAMETERS
    r_surface = 6371.2  # earths mean radius, often called a
    epoch_as_decimal = (1 - (12 - 12 / len(epochs)) / 12)

    # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
    # division by zero are ignored.
    Grs = []
    Brs = np.empty([0])
    tps = []
    for y in year_list:
        for e in epochs:
            [Br, theta, phi, r, indices] = load_epochs_r(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
                                                            truncation=truncation)
            if len(theta) == 300:
                if y > 2013:
                    # Only Gr is used so division by zero is ignored.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

                    # create list of Gr design matrices, one for each tp
                    Gr_swarm_template = Gr
                else:
                    [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)
                    Gr_champ_template = Gr

    # print(np.shape(Gr_champ_template))
    # print(np.shape(Gr_swarm_template))
    for t in grid:
        if t <= 2014:
            Grs.append(Gr_champ_template)
        else:
            Grs.append(Gr_swarm_template)

    return Grs


def matern_kernel(tp, tq, params):
    '''
    This function returns the covariance matrix resulting from the squared exponential covariance function (kernel)
    '''
    return (1 + np.sqrt(3) * np.abs(np.subtract.outer(tp, tq)) / params[0]) * np.exp((- np.sqrt(3) * np.abs(np.subtract.outer(tp, tq)) / params[0]))


def matern_kernel_element(tp, tq, params):
    '''
    This function returns the a single element of the covariance matrix resulting from the squared exponential
    covariance function (kernel)
    '''
    # return (1 + np.sqrt(3) * np.abs(tp - tq) / params[0]) * np.exp((- np.sqrt(3) * np.abs(tp - tq) / params[0]))
    return (1 + (np.sqrt(5) * np.abs(tp - tq)) / params[0] + (5 * (tp - tq)**2) / (3 * params[0]**2)) * \
            np.exp((- np.sqrt(5) * np.abs(tp - tq) / params[0]))
    # return np.exp(-1/(2*params[0]**2) * (tp - tq)**2)


# -------------------------------------------------- ADDITIONAL TOOLS ------------------------------------------------

def matprint(mat, fmt="g"):
    '''
    Prints 2-d arrays with equally spaced cells, for inspection.
    mat - 2d array to be printed
    fmt - formatting
    :return: prints 2 d array
    '''
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")