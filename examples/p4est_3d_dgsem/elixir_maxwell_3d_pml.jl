using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell

###############################################################################
# semidiscretization of the Maxwell equations with a uniaxial PML

equations = MaxwellEquations3D(UPML())

# twice the width of the tetrahedral case, which the hexahedra resolve
dipole = HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2,
                        GaussianPulse(0.4; delay = 1.4))
dipole_field = HertzianDipoleField(dipole, equations)

coordinates_min = (-1.5, -1.5, -1.5)
coordinates_max = (1.5, 1.5, 1.5)
pml_thickness = 0.5
pml_profile = PMLProfile(coordinates_min, coordinates_max, pml_thickness)

initial_condition = initial_condition_zero

boundary_condition = boundary_condition_silver_mueller
boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                       y_neg = boundary_condition, y_pos = boundary_condition,
                       z_neg = boundary_condition, z_pos = boundary_condition)

polydeg = 2
surface_flux = flux_upwind
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

source_terms = ProjectedSourceTerms(CombinedSourceTerms(SourceTermsPML(pml_profile),
                                                        dipole), equations, solver)

trees_per_dimension = (6, 6, 6)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 1,
                 periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions, source_terms)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 4.0)
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
