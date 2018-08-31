TOOLBOX
- for spherical harmonic analysis and Gaussian processes regression

Created as part of the BSc: "Modelling rapid changes in the Earth's magnetic field" by Eigil Yuichi Hyldgaard Lippert

Summary:
This toolbox contains function for building spatial and spatial temporal models based on the theory of spherical harmonics and Gaussian processes for regression. The directory contains two example main files, produces some of the output seen in the report on the subject. The main functions are contained in "functions.py" and some help tools are contained in the "GMT_tools.py"

List of functions in "functions.py":
# LOADING:
	load_single_epoch()
	load_single_VO()
	get_vo_indices()
	
# SINGLE EPOCH
	global_field_model()
	L1_norm()
	L2_norm()
	compute_G_cmb()
	L_curve_corner()

# TIME DEPENDENCY
	matern_kernel()
	matern_kernel_element()
	time_dep_synth_grid()
	design_SHA_GP()
	design_SHA_per_epoch()
	design_time_grid()
	
# PLOTTING TOOLS
	global_field_plot()
	time_dep_global_field_plot()
	errors_plot()
	L_curve_plot()
	L_curve_geometrical_corner()
	power_spectrum()

# ADDITIONAL TOOLS
	matprint()

List of functions in "GMT_tools.py":
	get_Pnm()
	design_SHA()
	synth_grid()
	mauersberger_lowes_spec()
