using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell
using StaticArrays: SVector
using LinearAlgebra: norm

###############################################################################
# semidiscretization of the Maxwell equations with a Hertzian dipole source

equations = MaxwellEquations3D()

# dipole along z at the origin; the time signal is wider than the spatial
# regularization so that the radiated field follows the point dipole closely
dipole = HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1,
                        GaussianPulse(0.4; delay = 1.4))
polydeg = 3
surface_flux = flux_upwind
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

source_terms = ProjectedSourceTerms(dipole, equations, solver)

# exact exterior field of the regularized dipole for comparison, see the tests
dipole_field = HertzianDipoleField(dipole, equations)

initial_condition = initial_condition_zero

boundary_condition = boundary_condition_silver_mueller
boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                       y_neg = boundary_condition, y_pos = boundary_condition,
                       z_neg = boundary_condition, z_pos = boundary_condition)

trees_per_dimension = (4, 4, 4)
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 1,
                 periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions, source_terms)

###############################################################################
# ODE solvers, callbacks etc.

# the pulse peak reaches the boundary at t = 2.4; until then the domain is free space
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[])
alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

# Adaptive mesh refinement: the Löhner indicator on the energy density follows
# the pulse, the distance indicator keeps the elements around the narrow
# source refined.
struct IndicatorDistance{RealT}
    center::SVector{3, RealT}
    radius::RealT
end

function (indicator::IndicatorDistance)(u, mesh, equations, dg, cache; kwargs...)
    (; node_coordinates) = cache.elements
    return map(Trixi.eachelement(dg, cache)) do element
        center = SVector{3}(ntuple(d -> sum(view(node_coordinates, d, :, :, :, element)) /
                                        length(view(node_coordinates, d, :, :, :,
                                                    element)), 3))
        max(0.0, 1 - norm(center - indicator.center) / indicator.radius)
    end
end

# The smoothness indicator is amplitude blind, so it is gated by the energy
# density; `wave_cap` limits the level the pulse can reach.
struct IndicatorSourceAndWave{Wave, Energy, Source, RealT}
    wave::Wave
    energy::Energy
    source::Source
    wave_cap::RealT
    energy_floor::RealT
end

function (indicator::IndicatorSourceAndWave)(u, mesh, equations, dg, cache; kwargs...)
    alpha_wave = indicator.wave(u, mesh, equations, dg, cache; kwargs...)
    alpha_energy = indicator.energy(u, mesh, equations, dg, cache; kwargs...)
    alpha_source = indicator.source(u, mesh, equations, dg, cache; kwargs...)
    return max.(min.(alpha_wave, indicator.wave_cap) .*
                (alpha_energy .> indicator.energy_floor), alpha_source)
end

amr_indicator = IndicatorSourceAndWave(IndicatorLöhner(semi, variable = energy_total),
                                       IndicatorMax(semi, variable = energy_total),
                                       IndicatorDistance(SVector(0.0, 0.0, 0.0), 0.45),
                                       0.45, 1.0e-5)
med_threshold = 0.2
max_threshold = 0.5
amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level = 1, med_level = 2, med_threshold,
                                      max_level = 3, max_threshold)
amr_interval = 25
amr_callback = AMRCallback(semi, amr_controller, interval = amr_interval,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks)
