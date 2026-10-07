using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell
using StaticArrays: SVector

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

# Adaptive mesh refinement with two levels: the pulse is followed with the
# smoothness indicator on the energy density, gated by an energy floor, and the
# elements along the injection box are always refined so that the interface
# never crosses a mortar.
struct IndicatorBoxAndWave{Wave, Energy, RealT}
    wave::Wave
    energy::Energy
    energy_floor::RealT
    half_width::RealT
    band::RealT
end

function (indicator::IndicatorBoxAndWave)(u, mesh, equations, dg, cache; kwargs...)
    alpha_wave = indicator.wave(u, mesh, equations, dg, cache; kwargs...)
    alpha_energy = indicator.energy(u, mesh, equations, dg, cache; kwargs...)
    (; node_coordinates) = cache.elements
    return map(Trixi.eachelement(dg, cache)) do element
        center = TrixiMaxwell.element_centroid(node_coordinates, element)
        on_box = abs(maximum(abs, center) - indicator.half_width) < indicator.band
        on_box ? 1.0 :
        alpha_wave[element] * (alpha_energy[element] > indicator.energy_floor)
    end
end

amr_indicator = IndicatorBoxAndWave(IndicatorLöhner(semi, variable = energy_total),
                                    IndicatorMax(semi, variable = energy_total),
                                    1.0e-5, 0.5, 0.25)
amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level = 1, med_level = 2, med_threshold = 0.2,
                                      max_level = 2, max_threshold = 0.5)
amr_interval = 20
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
