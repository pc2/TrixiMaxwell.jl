using OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiMaxwell
using StaticArrays: SVector

###############################################################################
# semidiscretization of the Maxwell equations with a dielectric half space

equations = MaxwellEquations3D(Heterogeneous())

epsilon_left = 1.0
epsilon_right = 4.0

@doc raw"""
    InitialConditionFresnel(epsilon_left, epsilon_right; width = 0.2, center = -0.5)

Gaussian pulse at normal incidence on the plane interface ``x = 0`` between two
dielectrics. For ``t \ge 0`` the exact solution is the incident pulse plus the
reflected pulse with amplitude ``r = (Z_2 - Z_1) / (Z_2 + Z_1)`` on the left
and the transmitted pulse with amplitude ``\tau = 2 Z_2 / (Z_1 + Z_2)`` and
width scaled by ``c_2 / c_1`` on the right. The material components of the
returned state follow the sign of ``x`` and are overwritten per element by
[`set_materials!`](@ref).
"""
struct InitialConditionFresnel{RealT}
    epsilon_left::RealT
    epsilon_right::RealT
    width::RealT
    center::RealT
end

function InitialConditionFresnel(epsilon_left, epsilon_right; width = 0.2, center = -0.5)
    return InitialConditionFresnel(promote(epsilon_left, epsilon_right, width, center)...)
end

function (initial_condition::InitialConditionFresnel)(x, t,
                                                      equations::MaxwellEquations3D{Heterogeneous})
    RealT = eltype(x)
    eps1 = convert(RealT, initial_condition.epsilon_left)
    eps2 = convert(RealT, initial_condition.epsilon_right)
    c1 = inv(sqrt(eps1))
    c2 = inv(sqrt(eps2))
    Z1 = inv(sqrt(eps1))
    Z2 = inv(sqrt(eps2))
    reflection = (Z2 - Z1) / (Z2 + Z1)
    transmission = 2 * Z2 / (Z1 + Z2)

    x0 = convert(RealT, initial_condition.center)
    width = convert(RealT, initial_condition.width)
    pulse(s) = exp(-(s / width)^2)

    if x[1] < 0
        incident = pulse(x[1] - c1 * t - x0)
        reflected = reflection * pulse(-x[1] - c1 * t - x0)
        Ey = incident + reflected
        Hz = (incident - reflected) / Z1
        eps = eps1
    else
        Ey = transmission * pulse(c1 / c2 * x[1] - c1 * t - x0)
        Hz = Ey / Z2
        eps = eps2
    end
    z = zero(Ey)

    return SVector(z, Ey, z, z, z, Hz, eps, one(RealT), z)
end
pulse_width = 0.2
initial_condition = InitialConditionFresnel(epsilon_left, epsilon_right;
                                            width = pulse_width)

function material_at(x)
    x[1] < 0 ? Material(epsilon = epsilon_left) :
    Material(epsilon = epsilon_right)
end

# the exact solution enters and leaves through the x faces
incident = BoundaryConditionIncidentField(initial_condition)
boundary_conditions = (; x_neg = incident, x_pos = incident)

polydeg = 3
surface_flux = flux_upwind
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux)

trees_per_dimension = (8, 2, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 coordinates_min = (-1.0, 0.0, 0.0), coordinates_max = (1.0, 0.5, 0.5),
                 initial_refinement_level = 0,
                 periodicity = (false, true, true))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.8)
ode = semidiscretize(semi, tspan)
set_materials!(ode.u0, semi, material_at)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

# Adaptive mesh refinement follows the pulse; the smoothness indicator is
# amplitude blind, so it is gated by the energy density. The mesh is not adapted
# before the run, so the materials set per element are kept.
struct IndicatorGated{Wave, Energy, RealT}
    wave::Wave
    energy::Energy
    energy_floor::RealT
end

function (indicator::IndicatorGated)(u, mesh, equations, dg, cache; kwargs...)
    alpha_wave = indicator.wave(u, mesh, equations, dg, cache; kwargs...)
    alpha_energy = indicator.energy(u, mesh, equations, dg, cache; kwargs...)
    return alpha_wave .* (alpha_energy .> indicator.energy_floor)
end

amr_indicator = IndicatorGated(IndicatorLöhner(semi, variable = energy_total),
                               IndicatorMax(semi, variable = energy_total), 1.0e-5)
amr_controller = ControllerThreeLevel(semi, amr_indicator;
                                      base_level = 0, med_level = 1, med_threshold = 0.2,
                                      max_level = 2, max_threshold = 0.5)
amr_interval = 20
amr_callback = AMRCallback(semi, amr_controller, interval = amr_interval,
                           adapt_initial_condition = false)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # overwritten by the stepsize callback
            ode_default_options()..., callback = callbacks)
