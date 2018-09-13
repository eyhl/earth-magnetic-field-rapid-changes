from lib import GMT_tools as gmt
from lib import functions as ft
import numpy as np

'''
This script computes a geomagnetic model in a single component for a single epoch. Optional regularisation.
'''
# ---------------------------------------------- CONSTANTS -----------------------------------------------------------
r_surface = 6371.2  # earths mean radius
r_core = 3480.  # earths mean core radius

# -------------------------------------------- INPUT CHOICES ---------------------------------------------------------
# choose data paths, separated by commas:
paths = ['data/VO_SWARM_MF_0101.txt',
         'data/VO_CHAMP_MF0101.txt']

err_path = 'data/VO_MF_SWARM_COV_diag_0104.txt'

# year between 2001-2017 (if both swarm and champ data is available), epoch choose 3,7 or 11:
[year, epoch] = [2017, 7]

# choose field component to compute:
Bi = 'Br'

# spherical harmonic degree:
degree = 20

# regularisation method, choose between 'L1' or 'L2' anything else will yield no regularisation:
reg_method = 'L1'

# In case of L1, create list of alphas. Number of alphas to be evaluated and the limits given as 10^limit:
if reg_method == 'L1':
    [spacing, alpha_min, alpha_max] = [50, -5, -1]
else:
    [spacing, alpha_min, alpha_max] = [100, -6.8, -6.7]

alpha_list = np.logspace(alpha_min, alpha_max, spacing)


# ---------------------------------------------------- RUN ---------------------------------------------------------
# LOAD
file = []
for path in paths:
    file.append(path)
[Br, Bt, Bp, theta, phi, r, errors] = ft.load_single_epoch(files=file, year=year, epoch=epoch, errors_path=err_path)

# PRE-MODELLING
# compute design matrix at core mantle boundary
[Gr_cmb, Gt_cmb, Gp_cmb] = ft.compute_G_cmb(stepsize=5, degree=degree)

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

# ------------------------------------------------ MODELLING ---------------------------------------------------------
if reg_method == 'L1':
    # compute models corresponding to different alphas
    [model_list, residuals_list, misfit_list, model_norm_list, gamma_list] = ft.L1_norm(Bi=Bi, Gi=Gi, L=Gr_cmb,
                                                                                        degree=degree,
                                                                                        alpha_list=alpha_list,
                                                                                        errors=errors[0],
                                                                                        gamma=1, eps=1e-4,
                                                                                        converged=1e-3, printall=False)

    # find the alpha that yields the best model:
    [best_alpha, alpha_index, kappa] = ft.L_curve_corner(rho=misfit_list, eta=model_norm_list, alpha=alpha_list)

    model_final = model_list[alpha_index]
    best_gammas = gamma_list[alpha_index]
    residuals = residuals_list[alpha_index]


elif reg_method == 'L2':
    # compute models corresponding to different alphas
    [model_list, residuals_list, misfit_list, model_norm_list] = ft.L2_norm(Bi=Bi, Gi=Gi, L=Gr_cmb,
                                                                            alpha_list=alpha_list, errors=None)

    # find the alpha that yields the best model:
    [best_alpha, alpha_index, kappa] = ft.L_curve_corner(rho=misfit_list, eta=model_norm_list, alpha=alpha_list)

    model_final = model_list[alpha_index]
    residuals = residuals_list[alpha_index]

else:
    [model_final, residuals, misfit_norm, model_norm] = ft.global_field_model(Bi=Bi, Gi=Gi, L=Gr_cmb, degree=degree,
                                                                              errors=None)

np.savetxt("model_vector_L1.txt", model_final)
np.savetxt("residuals_vector_test.txt", residuals)
