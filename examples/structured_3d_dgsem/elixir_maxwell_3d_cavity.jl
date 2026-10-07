using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations

equations = MaxwellEquations3D()

initial_condition = initial_condition_cavity

boundary_conditions = boundary_condition_perfect_electric_conductor

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_upwind)

cells_per_dimension = (8, 8, 8)
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by stepsize callback
            ode_default_options()..., callback = callbacks)
