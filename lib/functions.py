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
pd.options.mode.chained_assignment = None  # default='warn'


# ------------------------------------------------------ LOADING ------------------------------------------------------
def load_single_epoch(files, year, epoch, errors_path=None, header_size=4, NaN_marker=99999.):
    '''
    This functions loads in data from the global magnetic field for a chosen year and epoch (3,7,11).
    Data points should have following units:
    theta, phi - in degrees; r - in km; Br, Bt, Bp - in nT.

    Input:
    files         - the data files
    year          - chosen year, available: [2000 - 2010], [2013-2017]
    epoch         - chosen epoch, available: [3, 7, 11]
    add_errors    - defaulted False. Inpute data path point to error covariance matrix. Here only variance is assumed so
                         it is only the diagonal that is used.
    header_size   - defaulted to 4 for this project
    NaN_marker    - defaulted to 99999 in this project

    Output:
    Br, Bt, Bp    - field components, in nT
    theta, phi, r - spherical coordinates in radians, radians, km.
    errors        - list containing three vectors, holding the errors for Br, Bt and Bp respectively.


    The delimeter is defaulted to be whitespaces and the structure of the input file is assumed to be:
    [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    Values at the poles (theta = 0 and 180) are assumed erroneous and their rows are removed.
    NaN marker is set to default as 99999, and all rows including NaNs are removed.

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

    if errors_path is None:
        # empty errors object
        errors = None
    else:
        # Read in uncertainty estimates, corresponding to chosen year
        errors = pd.read_table(errors_path, header=None)

        # Extract diagonal (length 900, corresponding to 300 VOs in r, theta, phi)
        err_r = np.diag(errors)[0:300]
        err_t = np.diag(errors)[300:600]
        err_p = np.diag(errors)[600:900]

        # Add errors to existing data frame
        data_time.loc[:, 'err_r'] = err_r
        data_time.loc[:, 'err_t'] = err_t
        data_time.loc[:, 'err_p'] = err_p

    # removes pole data points, and NaNs
    data_time = data_time[data_time['theta'] != 0]  # drop rows with theta=0
    data_time = data_time[data_time['theta'] != 180]  # drop rows with theta=180
    data_time = data_time.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data_time = data_time.dropna(how='any')  # drop rows with NaNs

    # save grid variables based on data points
    Br = data_time['Br'].values
    Bt = data_time['Bt'].values
    Bp = data_time['Bp'].values
    theta = data_time['theta'].values * rad  # convert to radians
    phi = data_time['phi'].values * rad
    r = data_time['r'].values
    if errors is not None:
        errors = [data_time['err_r'].values, data_time['err_t'].values, data_time['err_p'].values]

    return Br, Bt, Bp, theta, phi, r, errors


def load_single_vo(files, Bi, theta, phi, header_size=4, NaN_marker=99999.):
    '''
    This functions loads in data for a single Virtual Observatory (VO), and computes the Secular Variation for as many
    epochs as possible, defined as the difference in the magnetic field between two epochs one year apart from each
    other. The delimeter in loaded file is defaulted to be whitespaces and the structure of the input file is assumed
    to be: [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    Data points should have following units: theta, phi [degrees], r [km] and (Br, Bt, Bp) [nT].
    Input:

    files        - the data files
    Bi           - the chosen component
    (theta, phi) - the coordinates of the VO in degrees.
    header_size  - defaulted to 4 for this project
    NaN_marker   - defaulted to 99999 in this project

    Values at the poles (theta = 0 and 180) are assumed erroneous and their rows are removed.
    NaN marker is set to default as 99999, and all rows including NaNs are removed.

    returns one vector per component, holding the corresponding magnetic field values. the "times" vector holds the
    quantified times, sv is a vector of the secular variation for the chosen magnetic component and sv_time is the
    time related to the secular variation.

    :return: Br, Bt, Bp, times, sv, sv_time
    '''
    # for conversion
    rad = np.pi / 180

    # for quantifying the epochs into decimals, e.g. 2014.3333 for 1st epoch of 2014.
    one_third_year = (1 - (2 / 3))

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
    data_VO = data[thetas & phis]

    # handles NaNs
    data_VO = data_VO.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data_VO.loc[data_VO.isnull().any(axis=1), 0:2] = np.nan
    data_VO.loc[data_VO.isnull().any(axis=1), 3::] = np.nan
    data_VO = data_VO.fillna(value=0)  # fill NaNs with zeros

    # every year appears several times in file
    years = np.unique(data_VO['Year'].values)

    sv_time = []
    sv = []
    # This loop takes epoch for year y and check if the same epoch for year y + 1 exists. If so it is used for sv.

    for y in years:
        df1 = data_VO.query('Year==' + str(y))
        df2 = data_VO.query('Year==' + str(y + 1))
        months1 = df1['Month'].values
        months2 = df2['Month'].values
        for idx, m in enumerate(months1):
            # try to look at epoch for next year, if year not existing it will continue
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

    # convert back to nans for easy removal
    data_VO = data_VO.replace(0, np.nan)  # set all 0's to NaN
    data_VO = data_VO.dropna(how='any')

    column_name = 'Month'
    mask_epoch3 = data_VO.Month == 3
    mask_epoch7 = data_VO.Month == 7
    mask_epoch11 = data_VO.Month == 11
    data_VO.loc[mask_epoch3, column_name] = one_third_year
    data_VO.loc[mask_epoch7, column_name] = 2 * one_third_year
    data_VO.loc[mask_epoch11, column_name] = 2.9999999 * one_third_year

    times = (data_VO['Year'] + data_VO['Month']).values

    # save data in seperate variables
    Br = data_VO['Br'].values
    Bt = data_VO['Bt'].values
    Bp = data_VO['Bp'].values

    return Br, Bt, Bp, times, sv, sv_time


def get_vo_indices(files, year, epoch, vo_coordinates, truncation=None, header_size=4, NaN_marker=99999.):
    '''
    This functions groups data set by Virtual Observatory (VO) coordinates. In order to work with one or several
    specific VO's it is useful to know their position(s) in the data set. Also returns the full list of thetas and phis,
    and could be called like theta[indices[i][j]]. An example set of coordinates:
    [(48.17388, 8.7721), (143.524, 86.00912), (95.96354000000001, 123.77911)]

    Input parameters:
    files          - list of data files
    year           - chosen year of data, available: [2000 - 2010], [2013-2017]
    epoch          - chosen epoch of data, available: [3, 7, 11]
    vo_coordinates - list of tuples containing coordinates for the wanted VO(s). e.g. [(75, 10), (180, 0)]
    truncation     - set to a number in order to truncate the data, e.g. 1 in order to only look at the first data point.
    header_size    - number of header rows in the data files
    NaN_marker     - change if data contains a different NaN marker than 99999.

    The delimeters of the files are defaulted to be whitespaces and the structure of the input file is assumed to be:
    [theta, phi, Year, Month, Time, r, Br, Bt, Bp, N_{data}]
    NaN marker is set to default as 99999, and all rows including NaNs are removed.
    note: theta and phi is returned in radians.
    :return: theta, phi, indices
    '''

    # conversion into radians
    rad = np.pi / 180

    # selecting chosen year and epoch
    y = year
    m = epoch
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
    data.loc[data['phi'] < 0, 'phi'] += 360  # note added +360, yields more stable B

    single_year_mask = data['Year'] == y
    single_epoch_mask = data['Month'] == m
    single_year = data[single_year_mask & single_epoch_mask].reset_index()
    g = single_year.groupby(['theta', 'phi'])

    # loop through input vo coordinates, and create list of indices.
    indices = []
    for coords in vo_coordinates:
        current_index = g.get_group(coords).index.values
        indices.append(current_index)

    # selects chosen data time (year, epoch)
    data = data.query(year)
    data = data.query(epoch)

    # removes Bt, Bp and the N_data columns
    data = data.drop(['Bt', 'Bp', 'N_{data}'], axis=1)
    data = data.reset_index()

    data = data.replace(NaN_marker, np.nan)  # set all 99999 values to NaN
    data = data.dropna(how='any')  # drop rows with NaNs

    # save grid variables based on data points
    Br = data['Br'].values
    theta = data['theta'].values * rad  # convert to radians
    phi = data['phi'].values * rad
    r = data['r'].values

    if truncation:
        k = truncation
        [theta, phi] = [theta[0:k], phi[0:k]]

    return theta, phi, indices


# ---------------------------------------------------- SINGLE EPOCH ---------------------------------------------------
def global_field_model(Bi, Gi, L, degree, errors=None, regularise='', alpha=1e-8, gamma=1, eps=1e-4):
    '''
    Computes global model of the geomagnetic field at the surface, given by least squares estimate:
        model = G.T.dot(W.dot(G))^(-1).G.T.dot(W.dot(d))
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

    :param errors:
    :return: model, residuals, misfit_norm, model_norm
    '''
    N = len(Bi)

    # figure out what which dimension of L that is not the "number of model coefficients"
    size_array = np.array([np.size(L, 0), np.size(L, 1)])  # array with number of rows and columns in L
    correct_size = int(size_array[np.argwhere(size_array != degree*(degree + 2))])  # size != number of coefficients

    # defining the weight matrix for L1
    wm = np.ones(correct_size) * 1/np.sqrt(gamma ** 2 + eps ** 2)  # Ekblom's measure, avoid division by zero

    # right hand side (rhs) are similar in all cases
    if errors is not None:
        rhs = Gi.T.dot(np.diag(errors).dot(Bi))
    else:
        rhs = Gi.T.dot(Bi)

    if regularise == 'L1':
        if errors is not None:
            lhs = (Gi.T.dot(np.diag(errors).dot(Gi)) + alpha**2 * L.T.dot(np.diag(wm)).dot(L))  # left hand side (lhs)
        else:
            lhs = (Gi.T.dot(Gi) + alpha**2 * L.T.dot(np.diag(wm)).dot(L))  # left hand side (lhs)

        model = np.linalg.solve(lhs, rhs)

        # L1-norm defined as sum(abs(gamma[i])):
        model_norm = 1/N * np.sum(np.abs(gamma))

    elif regularise == 'L2':
        if errors is not None:
            lhs = (Gi.T.dot(np.diag(errors).dot(Gi)) + alpha**2 * L.T.dot(L))  # left hand side (lhs)
        else:
            lhs = (Gi.T.dot(Gi) + alpha**2 * L.T.dot(L))  # left hand side (lhs)

        model = np.linalg.solve(lhs, rhs)

        # L2-norm defined as sum(L.dot(m)**2):
        model_norm = 1/N * np.sum(L.dot(model) ** 2)

    else:
        if errors is not None:
            lhs = Gi.T.dot(np.diag(errors).dot(Gi))  # left hand side (lhs)
        else:
            lhs = Gi.T.dot(Gi)  # left hand side (lhs)

        model = np.linalg.solve(lhs, rhs)
        model_norm = 0

    residuals = (Bi - Gi.dot(model))
    misfit_norm = 1/N * np.sqrt(residuals.T.dot(residuals))

    return model, residuals, misfit_norm, model_norm


def L1_norm(Bi, Gi, L, degree, alpha_list, errors=None, gamma=1, eps=1e-4, converged=0.001, printall=False):
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

    :param errors:
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
        [model_previous, dummy, dummy, dummy] = global_field_model(Bi=Bi, Gi=Gi, L=L, degree=degree, errors=None,
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
            if errors is not None:
                rhs = Gi.T.dot(np.diag(errors).dot(Bi))
                lhs = (Gi.T.dot(np.diag(errors).dot(Gi)) + alpha**2 * L.T.dot(np.diag(wm)).dot(L))
            else:
                rhs = Gi.T.dot(Bi)
                lhs = (Gi.T.dot(Gi) + alpha ** 2 * L.T.dot(np.diag(wm)).dot(L))

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


def L2_norm(Bi, Gi, L, alpha_list, errors=None):
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

    :param errors:
    :return: model_list, residuals_list, misfit_list, model_norm_list
    '''
    # for normalisation of model and misfit norm
    N = len(Bi)

    model_list = []
    misfit_list = []  # sometimes denoted rho
    model_norm_list = []  # sometimes denoted eta
    residuals_list = []

    for alpha in alpha_list:
        if errors is not None:
            rhs = Gi.T.dot(np.diag(errors).dot(Bi))
            lhs = Gi.T.dot(np.diag(errors).dot(Gi)) + alpha**2 * L.T.dot(L)
        else:
            rhs = Gi.T.dot(Bi)
            lhs = Gi.T.dot(Gi) + alpha ** 2 * L.T.dot(L)

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

    corner_index = np.argmax(kappa)
    corner_alpha = alpha[corner_index]  # the optimum alpha as found in the L curve corner

    return corner_alpha, corner_index, kappa


# -------------------------------------------------- TIME DEPENDENCY -------------------------------------------------

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

    B_theta = 1
    B_phi = 1

    return (B_r, B_theta, B_phi)


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

    # checks dimensionality of input design matrix
    single_time = False
    if np.array(Gi_t).ndim != 3:
        single_time = True

    # Creates the k* matrix as defined in Rasmussen and Williamson eq. 2.25 combined with spherical harmonic design
    # matrix to obtain spatial-temporal design matrix called A.
    A = np.empty([0, n_coefficients * n_ref_times])
    for tp_idx, tp_val in enumerate(prediction_times):
        # sets the current design matrix corresponding to the number of times used.
        if single_time:
            current_design_matrix = Gi_t
        else:
            current_design_matrix = Gi_t[tp_idx]

        # for holding the complete k* matrix at a given tp
        cov_matrix_tp = np.empty([np.size(current_design_matrix, axis=0), 0])

        for coeff_count, gr_col in enumerate(current_design_matrix.T):
            # for temporarily storing the current coefficient times tq.
            ref_time_columns = np.empty([np.size(current_design_matrix, axis=0), 0])

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

            cov_matrix_tp = np.hstack((cov_matrix_tp, ref_time_columns))

        # stacks for each tp.
        A = np.vstack((A, cov_matrix_tp))

    return A


def design_SHA_GP_test(Gi_t, Gi_t_cmb, prediction_times, reference_times, tau, degree):
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

    # checks dimensionality of input design matrix
    single_time = False

    if np.array(Gi_t).ndim == 2:
        single_time = True

    # Creates the k* matrix as defined in Rasmussen and Williamson eq. 2.25 combined with spherical harmonic design
    # matrix to obtain spatial-temporal design matrix called A.
    A = np.empty([0, n_coefficients * n_ref_times])
    A_cmb = np.empty([0, n_coefficients * n_ref_times])
    for tp_idx, tp_val in enumerate(prediction_times):
        # sets the current design matrix corresponding to the number of times used.
        if single_time:
            current_design_matrix = Gi_t
            current_design_cmb = Gi_t_cmb
        else:
            current_design_matrix = Gi_t[tp_idx]
            current_design_cmb = Gi_t_cmb[tp_idx]

        # for holding the complete k* matrix at a given tp
        cov_matrix_tp = np.empty([np.size(current_design_matrix, axis=0), 0])

        for coeff_count, gr_col in enumerate(current_design_matrix.T):
            # for temporarily storing the current coefficient times tq.
            ref_time_columns = np.empty([np.size(current_design_matrix, axis=0), 0])

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

            cov_matrix_tp = np.hstack((cov_matrix_tp, ref_time_columns))

        A = np.vstack((A, cov_matrix_tp))

        if tp_idx == 0:
            cov_cmb_tp = np.empty([np.size(current_design_cmb, axis=0), 0])
            for coeff_count, gr_col in enumerate(current_design_cmb.T):
                # for temporarily storing the current coefficient times tq.
                ref_time_columns = np.empty([np.size(current_design_cmb, axis=0), 0])

                # Determines if the degree has increased, based on the current order:
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

                cov_cmb_tp = np.hstack((cov_cmb_tp, ref_time_columns))

            # stacks for each tp.
            A_cmb = np.vstack((A_cmb, cov_cmb_tp))

    return A, A_cmb


def design_SHA_per_epoch(data_paths, year_list, epochs, degree, truncation=None):
    ''' This function loads data, and creates a list of design matrices, field values and quantified times.
    An element in the list corresponds to a time.

    Inputs:
    data_paths          =  list of paths to data
    year_list           =  list years in question
    epochs              =  value for given epochs
    degree              =  spherical harmonic degree
    truncation          =  mainly for testing, enables truncating the data input

    output: Gs, tps, Brs, indices
    '''

    # HYPER-PARAMETERS
    r_surface = 6371.2  # earths mean radius, often called a
    r_core = 3480.

    epoch_as_decimal = (1 - (12 - 12 / len(epochs)) / 12)
    stepsize = 5

    theta_cmb = np.arange(180 - stepsize / 2, 0, - stepsize)
    phi_cmb = np.arange(-180 + stepsize / 2, 180 - stepsize / 2, stepsize)
    theta_cmb_grid, phi_cmb_grid = np.meshgrid(theta_cmb, phi_cmb)

    theta_cmb_grid = np.reshape(theta_cmb_grid, np.size(theta_cmb_grid), 1)
    phi_cmb_grid = np.reshape(phi_cmb_grid, np.size(phi_cmb_grid), 1)
    length = len(theta_cmb_grid)

    # comutes design matrix at core mantle boundary (cmb), based on grid.
    [Gr_cmb, Gt_cmb, Gp_cmb] = gmt.design_SHA(r_core / r_surface * np.ones(length),
                                              theta_cmb_grid, phi_cmb_grid, degree)


    # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
    # division by zero are ignored.
    Gs = []
    Grs_cmb = []
    Bs = np.empty([0])
    tps = []
    for y in year_list:
        for e in epochs:
            [Br, Bt, Bp, theta, phi, r, errors] = load_single_epoch(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
                                                            errors_path=None)

            n_data = len(Br) + len(Bt) + len(Bp)
            if n_data == 894:
                # Only Gr is used so division by zero is ignored.
                with np.errstate(divide='ignore', invalid='ignore'):
                    [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

                # create list of G design matrices, one for each tp
                G = np.vstack((Gr, Gt, Gp))
                Gs.append(G)
                Grs_cmb.append(Gr_cmb)

                # stack Bs for residuals
                B = np.hstack((Br, Bt, Bp))
                Bs = np.hstack((Bs, B))

                # quantifying tp and append to list.
                if e == 3:
                    tps.append(y + epoch_as_decimal)
                elif e == 7:
                    tps.append(y + 2 * epoch_as_decimal)
                else:
                    tps.append(y + 2.999999 * epoch_as_decimal)

    return Gs, Grs_cmb, np.array(tps), Bs


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
    Gs = []
    Brs = np.empty([0])
    tps = []
    for y in year_list:
        for e in epochs:
            [Br, Bt, Bp, theta, phi, r, errors] = load_single_epoch(files=[data_paths[0], data_paths[1]], year=y, epoch=e,
                                                            errors_path=None)

            n_data = len(Br) + len(Bt) + len(Bp)
            if n_data == 894:
                if y > 2013:
                    # Only Gr is used so division by zero is ignored.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

                    # create list of Gr design matrices, one for each tp
                    G_swarm_template = np.vstack((Gr, Gt, Gp))

                else:
                    [Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)
                    G_champ_template = np.vstack((Gr, Gt, Gp))

    for t in grid:
        if t <= 2014:
            Gs.append(G_champ_template)
        else:
            Gs.append(G_swarm_template)

    return Gs


# --------------------------------------------------- PLOTTING TOOLS -------------------------------------------------
def global_field_plot(model, model_name='not-specified', radius=6371.2, save=False, vmin=-1e6, vmax=1e6, component='r',
                      step=.25, projection_type='hammer', cmap='PuOr_r', lat0=0, lon0=0, polar_plots=True):
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

    :param polar_plots:
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

    if polar_plots:
        # Add polar basemap plots
        # North Pole:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.2, wspace=0.1)

        # Basemap object
        mn = Basemap(projection='nplaea', boundinglat=50, lon_0=0, resolution='l', round=True, ax=ax[0])

        # Create grid
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        xx, yy = mn(phi_grid, theta_bound[1] - theta_grid)

        # Colormesh interpolation
        mn.pcolormesh(xx, yy, B_i, cmap=cmap, norm=MidpointNormalize(midpoint=0.), vmin=vmin, vmax=vmax)
        mn.drawcoastlines(linewidth=1.)
        mn.drawparallels(np.arange(0, 90, 10), labels=[1, 1, 1, 1])

        title_string = ('North Pole')
        ttl = ax[0].title
        ttl.set_position([.5, 1.05])
        ax[0].set_title(title_string, fontsize=16, fontweight='bold')

        # South Pole
        # TODO will fix south pole projection in some way.
        # Basemap object
        ms = Basemap(projection='splaea', boundinglat=-50, lon_0=0, resolution='l', round=True, ax=ax[1])

        # Create grid
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        xx, yy = ms(phi_grid, theta_bound[1] - theta_grid)

        # outside = (xx < ms.xmin) | (xx > ms.xmax) | (yy < ms.ymin) | (yy > ms.ymax)
        # B_i = np.ma.masked_where(outside, B_i)

        # Colormesh interpolation
        ms.pcolormesh(xx, yy, B_i, latlon=False, cmap=cmap, norm=MidpointNormalize(midpoint=0.), vmin=vmin, vmax=vmax)
        ms.drawcoastlines(linewidth=1.)
        ms.drawparallels(np.arange(-90, 0, 10), labels=[1, 1, 1, 1])

        title_string = ('South Pole - not working yet')
        ttl = ax[1].title
        ttl.set_position([.5, 1.05])
        ax[1].set_title(title_string, fontsize=16, fontweight='bold')

        plt.show()


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

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    xx, yy = m(phi_grid, theta_bound[1] - theta_grid)
    m.pcolormesh(xx, yy, B_i, cmap=cmap, norm=MidpointNormalize(midpoint=0.), vmin=vmin, vmax=vmax)
    m.drawcoastlines(linewidth=0.25)
    m.drawparallels(np.arange(-80, 81, 20), labels=[1, 0, 0, 0])

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

    if point:
        ax.plot(misfit_list[alpha_index], model_norm_list[alpha_index], 'ro', label='alpha^2')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Misfit 2-norm, log-scale')
    ax.set_ylabel('Model norm, log-scale')
    #
    ax.grid(True, which="both", ls="-")
    ax.minorticks_on()

    ax.set_title('L-curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()

    return


def power_spectrum(model, ratio, degree, plot=True):
    '''
    Computes the Mauersberger-Lowes power spectrum, based on model parameters
    model   = array or list of model coefficients
    ratio   = ratio given like a/r, where a is the reference radius and r the radius of interest.
    degree  = maximum degree of the model.
    :return: wn, a vector of power sepctrum values at different degrees
    '''
    # initialisation
    wn = []

    # counts the current coefficient order
    c = 0

    #initial value for the index
    idx = 0
    while c <= degree * (degree + 2):

        # determines the degree based on the current order
        n = -1 + np.sqrt(1 + c)

        # check if current degree is integer
        is_integer = (n - int(n))

        # if the current degree is a integer, then add the squared coefficients to wn[i]
        if is_integer == 0 and n > 0:
            if int(n) == 1:
                wn.append((n + 1) * ratio ** (2 * n + 4) * np.sum(model[idx:3] ** 2))
                idx = 3
            else:
                wn.append((n + 1) * ratio ** (2 * n + 4) * np.sum(model[idx:int(c - 1)] ** 2))
                idx = int(c - 1)
        c += 1

    if plot:
        degree_range = np.arange(1, degree + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.semilogy(degree_range, wn, 'r-', label='model')

        plt.legend(fontsize=14)
        plt.title('Mauersberger-Lowes Power Spectrum', fontsize=22, fontweight='bold')
        plt.xlabel(r'$\bf{Spherical\ Harmonic\ Degree,\ \it{n}}$', fontsize=18, fontweight='bold')
        plt.ylabel(r'$\bf{R_n}$' + r'$,\ [(nT)^2]$', fontsize=18, fontweight='bold')
        ax.set_xticks(range(0, 21))
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='both', labelsize=18)
        plt.grid()
        plt.show()

    return wn

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


# ------------------------------------------------------ OUTDATED ----------------------------------------------------
# TODO skal slettes da jeg ikke fik regularising til at virke:
def design_time_grid_cmb(grid, reference_times, time_scale, degree, stepsize=5):
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
    r_core = 3480.

    # Computes Gr based on r, theta, phi grid. Only r-component is considered, hence runtime-errors from phi component
    # division by zero are ignored.
    A_cmb = []
    tps = []

    # computes a regular grid
    theta_cmb = np.arange(180 - stepsize / 2, 0, - stepsize)
    phi_cmb = np.arange(-180 + stepsize / 2, 180 - stepsize / 2, stepsize)
    theta_cmb_grid, phi_cmb_grid = np.meshgrid(theta_cmb, phi_cmb)

    theta_cmb_grid = np.reshape(theta_cmb_grid, np.size(theta_cmb_grid), 1)
    phi_cmb_grid = np.reshape(phi_cmb_grid, np.size(phi_cmb_grid), 1)
    length = len(theta_cmb_grid)

    print(np.shape(theta_cmb_grid))

    # comutes design matrix at core mantle boundary (cmb), based on grid.
    [Gr_cmb, Gt_cmb, Gp_cmb] = gmt.design_SHA(r_core / r_surface * np.ones(length),
                                              theta_cmb_grid, phi_cmb_grid, degree)
    # print(np.shape(Gr_cmb))
    # print(np.shape(reference_times))
    # print(np.shape(grid))
    # A_cmb = np.empty([0, np.size(Gr_cmb, axis=1) * len(reference_times)])
    # for t in grid:
    #     print(t)
    #     tmp_cmb = design_SHA_GP(Gi_t=Gr_cmb, prediction_times=[t], reference_times=reference_times, tau=time_scale,
    #                             degree=degree)
    #
    #     # print(np.shape(tmp_cmb))
    #     # A_cmb.append(tmp_cmb)
    #     A_cmb = np.vstack((A_cmb, tmp_cmb))
    # print(np.shape(A_cmb))
    tmp_cmb = design_SHA_GP(Gi_t=Gr_cmb, prediction_times=[t], reference_times=reference_times, tau=time_scale,
                                degree=degree)
    return A_cmb
