from lib import GMT_tools as gmt
from lib import functions as ft
import numpy as np

'''
Computes the forward problem given a model. Default is r-component, can be changed in "Forward problem"-section if
so wished for. 
'''
# ---------------------------------------------- CONSTANTS -----------------------------------------------------------
r_surface = 6371.2  # earths mean radius, often called a
r_core = 3480.  # earths mean core radius

# -------------------------------------------- INPUT CHOICES --------------------------------------------------------
# choose model path:
model_final = np.loadtxt("coefficients_L1.txt")

# choose data paths, separated by commas:
paths = ['data/VO_SWARM_MF_0101.txt',
         'data/VO_CHAMP_MF0101.txt']
file = []
for path in paths:
    file.append(path)

# choose year and epoch corresponding to the epoch which the model is based on.
[year, epoch] = [2017, 7]


# --------------------------------------------- FORWARD PROBLEM ------------------------------------------------------
# determines degree from number of coefficients
degree = int(np.sqrt(1 + len(model_final)) - 1)

# load field data, and compute design matrix
[Br, Bt, Bp, theta, phi, r, errors] = ft.load_single_epoch(files=file, year=year, epoch=epoch, errors_path=None)
[Gr, Gt, Gp] = gmt.design_SHA(r / r_surface, theta, phi, degree)

# Forward problem - generates synthetic data.
synthetic_data = Gr.dot(model_final)

residuals = Br - synthetic_data


# -------------------------------------------------- PLOTTING -------------------------------------------------------
ft.power_spectrum(model_final, r_surface/r_core, degree)
ft.errors_plot(residuals=residuals)
ft.errors_plot(residuals=residuals, choice=[False, True], latitude=theta)
ft.global_field_plot(model_final, model_name="Core field", radius=r_core, save='n', vmin=-1e6, vmax=1e6, cmap='PuOr_r')

