using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell
using Gmsh

###############################################################################
# semidiscretization of the Maxwell equations: extinction spectrum of a gold
# nanosphere in vacuum

# one inactive pole of each kind as the default, the sphere activates them
equations = MaxwellEquations3D(Heterogeneous(); drude = (DrudePole(0.0, 0.0),),
                               lorentz = (LorentzPole(0.0, 0.0, 0.0),))

# lengths in units of 80 nm: the sphere of radius 0.5 is 40 nm, a normalized
# frequency f is the vacuum wavelength 80 nm / f
length_unit = 80e-9
sphere_radius = 0.5
sphere_material = material_gold(length_unit)
material_at(x) = sum(abs2, x) < sphere_radius^2 ? sphere_material : Material()

# Gaussian plane-wave pulse along z, polarized in x, delayed so that it starts
# outside the total-field box
# modulated pulse covering 420 to 800 nm without a static component
incident_field = PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                           ModulatedGaussianPulse(0.145, 3.0; delay = 12.0))

initial_condition = initial_condition_zero

# the vacuum wavelengths are 5 to 10 times the sphere radius
polydeg = 2
surface_flux = flux_upwind
solver = DGMulti(polydeg = polydeg,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm())

# sphere r = 0.5 inside a TFSF box of half-side 0.9 and an absorbing sphere r = 2.5
# mesh_order = 2 curves the elements onto the sphere, 1 keeps them straight-sided
size_factor = 3.0
mesh_order = 2
mesh_file = download_mesh("3D_RCS_SGBC_Sphere_Box.geo")
imported_mesh = read_gmsh(mesh_file; size_factor, order = mesh_order)
mesh = DGMultiMesh(solver, imported_mesh)

is_total_field(x) = all(abs.(x) .< 0.9)
tfsf = TotalFieldScatteredField(incident_field, mesh, is_total_field)

boundary_conditions = (; SMA = boundary_condition_silver_mueller, tfsf = tfsf)

source_terms = source_terms_dispersive

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions, source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# the plasmon has rung down by the end
tspan = (0.0, 65.0)
ode = semidiscretize(semi, tspan)
set_materials!(ode.u0, semi, material_at)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# vacuum wavelengths from 800 nm down to 420 nm
frequencies = range(0.1, 0.19, length = 19)
cross_section_callback = CrossSectionCallback(semi, tfsf, frequencies)

save_vtk_callback = SaveVtkCallback(dt = 0.5, output_directory = "out",
                                    filename = "gold_nanosphere")

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        cross_section_callback, save_vtk_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks)

sigma = cross_sections(cross_section_callback)
