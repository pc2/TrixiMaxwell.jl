using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations with a total-field/scattered-field box

equations = MaxwellEquations3D()

# Gaussian plane-wave pulse along x, polarized in z, delayed so that it starts
# outside the total-field box
incident_field = PlaneWave((1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                           GaussianPulse(0.25; delay = 1.5))

initial_condition = initial_condition_zero

is_total_field(x) = all(abs.(x) .< 0.5)
surface_flux = FluxTotalFieldScatteredField(flux_upwind, incident_field, is_total_field)

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

trees_per_dimension = (4, 4, 4)
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 1,
                 periodicity = false)

boundary_condition = boundary_condition_silver_mueller
boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                       y_neg = boundary_condition, y_pos = boundary_condition,
                       z_neg = boundary_condition, z_pos = boundary_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])
alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks)
