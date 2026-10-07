using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell
using Gmsh

###############################################################################
# semidiscretization of the Maxwell equations: scattering cross section of a
# dielectric sphere

equations = MaxwellEquations3D(Heterogeneous())

sphere_radius = 0.5
sphere_material = Material(epsilon = 2.25)
material_at(x) = sum(abs2, x) < sphere_radius^2 ? sphere_material : Material()

# Gaussian plane-wave pulse along z, polarized in x, delayed so that it starts
# outside the total-field box
incident_field = PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                           GaussianPulse(0.15; delay = 1.6))

initial_condition = initial_condition_zero

polydeg = 3
surface_flux = flux_upwind
solver = DGMulti(polydeg = polydeg,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm())

# sphere r = 0.5 inside a TFSF box of half-side 0.9 and an absorbing sphere r = 2.5
# mesh_order = 2 curves the elements onto the sphere, 1 keeps them straight-sided
size_factor = 2.0
mesh_order = 2
mesh_file = download_mesh("3D_RCS_SGBC_Sphere_Box.geo")
imported_mesh = read_gmsh(mesh_file; size_factor, order = mesh_order)
mesh = DGMultiMesh(solver, imported_mesh)

is_total_field(x) = all(abs.(x) .< 0.9)
tfsf = TotalFieldScatteredField(incident_field, mesh, is_total_field)

boundary_conditions = (; SMA = boundary_condition_silver_mueller, tfsf = tfsf)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0)
ode = semidiscretize(semi, tspan)
set_materials!(ode.u0, semi, material_at)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# size parameter k a = pi f
frequencies = range(0.2, 1.4, length = 13)
cross_section_callback = CrossSectionCallback(semi, tfsf, frequencies)

save_vtk_callback = SaveVtkCallback(dt = 0.25, output_directory = "out",
                                    filename = "mie")

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
