from lib import GMT_tools as gmt
from lib import functions as ft
import numpy as np

'''
This main script computes a geomagnetic model. User can choose different regularisation methods and other options.
All mandatory input is put into "INPUT CHOICES" section, 
For custom options see "OPTIONAL CHOICES" section. 
'''
# ---------------------------------------------- CONSTANTS -----------------------------------------------------------
r_surface = 6371.2  # earths mean radius, often called a
r_core = 3480.  # earths mean core radius

# -------------------------------------------- INPUT CHOICES ---------------------------------------------------------
# choose data paths, separated by commas:
paths = ['data/VO_SWARM_MF_0101.txt',
         'data/VO_CHAMP_MF0101.txt']

# year between 2001-2017 (if both swarm and champ data is available), epoch choose 3,7 or 11:
[year, epoch] = [2014, 7]

# spherical harmonic degree:
degree = 20

# regularisation method, choose between 'L1' or 'L2' anything else will yield no regularisation:
reg_method = 'L1'

# number of alphas to be evaluated and the limits given as 10^limit:
if reg_method == 'L1':
    [spacing, alpha_min, alpha_max] = [50, -5, -1]
else:
    [spacing, alpha_min, alpha_max] = [100, -6.8, -6.7]

alpha_list = np.logspace(alpha_min, alpha_max, spacing)

# stepsize for the grid used to compute the G matrix at core mantle boundary
stepsize = 5


# -------------------------------------------- OPTIONAL CHOICES ------------------------------------------------------
# (see function strings for explanation)

# PRINT/PLOT OPTIONS:
# if true prints the first five model coefficients and the alpha (if available) used at this model.
printmodel = True

# set to False if a specific plot is not wanted:
[L_curve, error_hist, error_lat, global_field_map] = [True, True, True, True]

# evaluate model at radius, r_input. Choose cmap=map_color; e.g. PuOr_r or tab20b. Save figure yes/no; 'y' or 'n'
map_color, save = ['PuOr_r', 'n']

# MAGNETIC FIELD OPTIONS:
# choose field component to compute:
Bi = 'Br'

# evaluate field map plot at radius:
r_input = r_core

# colorbar min and max, should correspond to magnetic min/max values at surface, core or another radius:
[vmin, vmax] = [-1e6, 1e6]

# L1 norm options:
[gamma, eps, conv, print_opt] = [1, 1e-4, 1e-3, False]


# ---------------------------------------------------- RUN ---------------------------------------------------------
# LOAD
file = []
for path in paths:
    file.append(path)
[Br, Bt, Bp, theta, phi, r] = ft.load_single_epoch(files=file, year=2014, epoch=7)

# PRE-MODELLING
# compute design matrix at core mantle boundary
[Gr_cmb, Gt_cmb, Gp_cmb] = ft.compute_G_cmb(stepsize, degree)

# compute design matrix
[Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

# chosen field component
if Bi == 'Br':
    Bi = Br
    Gi = Gr
elif Bi == 'Bt':
    Bi = Bt
    Gi = Gt
elif Bi == 'Bp':
    Bi = Bp
    Gi = Gp


# MODELLING
if reg_method == 'L1':
    # compute models corresponding to different alphas
    [model_list, residuals_list, misfit_list, model_norm_list, gamma_list] = ft.L1_norm(Bi=Bi, Gi=Gi, L=Gr_cmb,
                                                                                        degree=degree,
                                                                                        alpha_list=alpha_list,
                                                                                        gamma=gamma, eps=eps,
                                                                                        converged=conv,
                                                                                        printall=print_opt)

    # find the alpha that yields the best model:
    [best_alpha, alpha_index, kappa] = ft.L_curve_corner(rho=misfit_list, eta=model_norm_list, alpha=alpha_list)

    model_final = model_list[alpha_index]
    best_gammas = gamma_list[alpha_index]
    residuals = residuals_list[alpha_index]

    if printmodel:
        print('final model ', model_final[0:5])
        print('final alpha ', best_alpha)


elif reg_method == 'L2':
    # compute models corresponding to different alphas
    [model_list, residuals_list, misfit_list, model_norm_list] = ft.L2_norm(Bi=Bi, Gi=Gi, L=Gr_cmb,
                                                                                        alpha_list=alpha_list)

    # find the alpha that yields the best model:
    [best_alpha, alpha_index, kappa] = ft.L_curve_corner(rho=misfit_list, eta=model_norm_list, alpha=alpha_list)

    model_final = model_list[alpha_index]
    residuals = residuals_list[alpha_index]

    if printmodel:
        print('final model ', model_final[0:5])
        print('final alpha ', best_alpha)

else:
    [model_final, residuals, misfit_norm, model_norm] = ft.global_field_model(Bi=Bi, Gi=Gi, L=Gr_cmb, degree=degree)

    if printmodel:
        print('final model ', model_final[0:5])

# -------------------------------------------------- PLOTTING -------------------------------------------------------
if reg_method == 'L1' or reg_method == 'L2':
    if L_curve:
        ft.L_curve_plot(misfit_list=misfit_list, model_norm_list=model_norm_list, alpha_index=alpha_index)
    if error_hist:
        ft.errors_plot(residuals=residuals)
    if error_lat:
        ft.errors_plot(residuals=residuals, choice=[False, True], latitude=theta)
    if global_field_map:
        string = reg_method + '-norm'
        ft.global_field_plot(model_final, model_name=string, radius=r_input, cmap=map_color, save=save,
                             vmin=vmin, vmax=vmax)

else:
    if error_hist:
        ft.errors_plot(residuals=residuals)
    if error_lat:
        ft.errors_plot(residuals=residuals, choice=[False, True], latitude=theta)
    if global_field_map:
        string = 'Non-regularised'
        ft.global_field_plot(model_final, model_name=string, radius=r_input, cmap='PuOr_r', save=save,
                             vmin=vmin, vmax=vmax)

