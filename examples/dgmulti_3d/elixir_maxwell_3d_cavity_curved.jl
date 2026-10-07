using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations

equations = MaxwellEquations3D()

initial_condition = initial_condition_cavity

boundary_conditions = (; entire_boundary = boundary_condition_perfect_electric_conductor)

# set to source_terms_conductivity for a lossy medium, together with sigma > 0
source_terms = nothing

polydeg = 3
surface_flux = flux_upwind

solver = DGMulti(polydeg = polydeg,
                 element_type = Tet(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralWeakForm())

# Smooth warp of the interior of [-1, 1]^3 that vanishes on the faces, so the
# cube and its exact cavity mode are unchanged while all elements are curved.
function mapping(x, y, z)
    amplitude = 0.05
    warp = amplitude * sinpi(x) * sinpi(y) * sinpi(z)
    return (x + warp, y + warp, z + warp)
end

cells_per_dimension = (4, 4, 4)
mesh = DGMultiMesh(solver, cells_per_dimension, mapping)

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
