using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations in a cavity filled with a Drude medium

# lossless Drude pole; the cavity mode oscillates at sqrt(omega_0^2 + omega_p^2)
plasma_frequency = 2 * pi
equations = MaxwellEquations3D(drude = (DrudePole(plasma_frequency, 0.0),))

initial_condition = initial_condition_cavity

boundary_conditions = (; entire_boundary = boundary_condition_perfect_electric_conductor)

source_terms = source_terms_dispersive

polydeg = 3
surface_flux = flux_upwind

solver = DGMulti(polydeg = polydeg,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm())

cells_per_dimension = (4, 4, 4)
mesh = DGMultiMesh(solver, cells_per_dimension;
                   coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions, source_terms)

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
