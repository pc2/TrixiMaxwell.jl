struct Homogeneous end
struct Heterogeneous end
const MaterialModel = Union{Homogeneous, Heterogeneous}

struct NoPML end
struct UPML end
const AbsorberModel = Union{NoPML, UPML}

@doc raw"""
    MaxwellEquations3D(material = Homogeneous(), absorber = NoPML();
                       epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))
    MaxwellEquations3D(UPML(); epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))

Maxwell's curl equations for the fields ``(E, H)`` in a linear medium,
```math
\epsilon \partial_t E = \nabla \times H - \sigma E, \qquad
\mu \partial_t H = -\nabla \times E.
```
Units are normalized: `epsilon` and `mu` are relative values, the vacuum speed
of light and impedance are one, and a conductivity ``\sigma`` in SI units
becomes ``\sigma Z_0 L`` for the length unit ``L``.

The material model is a type parameter. With `Homogeneous`, the default, the
state is `(Ex, Ey, Ez, Hx, Hy, Hz)` and the material lives in the struct. With
`Heterogeneous` the state carries the passive components `epsilon`, `mu`,
`sigma` per node; they have zero flux and are set per element with
[`set_materials!`](@ref), the struct values being the defaults written by
initial conditions. `sigma` only acts when [`source_terms_conductivity`](@ref)
is passed as the source term of the semidiscretization.

With `UPML` the state additionally carries the auxiliary fields
`(px, py, pz, qx, qy, qz)` of the uniaxial perfectly matched layer, with zero
flux and zero initial value. They are driven by `SourceTermsPML` inside the
layer and stay zero elsewhere.
"""
struct MaxwellEquations3D{Material, Absorber, NVARS, RealT <: Real} <:
       Trixi.AbstractMaxwellEquations{3, NVARS}
    epsilon::RealT
    mu::RealT
    sigma::RealT
    impedance::RealT
    admittance::RealT
    speed_of_light::RealT

    function MaxwellEquations3D{Material, Absorber, NVARS}(epsilon, mu,
                                                           sigma) where {Material, Absorber,
                                                                         NVARS}
        impedance = sqrt(mu / epsilon)
        epsilon, mu, sigma, impedance, admittance, speed_of_light = promote(epsilon, mu,
                                                                            sigma,
                                                                            impedance,
                                                                            inv(impedance),
                                                                            inv(sqrt(epsilon *
                                                                                     mu)))
        return new{Material, Absorber, NVARS, typeof(epsilon)}(epsilon, mu, sigma,
                                                               impedance,
                                                               admittance, speed_of_light)
    end
end

num_material_components(::Type{Homogeneous}) = 0
num_material_components(::Type{Heterogeneous}) = 3

num_pml_components(::Type{NoPML}) = 0
num_pml_components(::Type{UPML}) = 6

function MaxwellEquations3D(material::Material = Homogeneous(),
                            absorber::Absorber = NoPML();
                            epsilon = 1.0, mu = one(epsilon),
                            sigma = zero(epsilon)) where {Material <: MaterialModel,
                                                          Absorber <: AbsorberModel}
    NVARS = 6 + num_material_components(Material) + num_pml_components(Absorber)
    return MaxwellEquations3D{Material, Absorber, NVARS}(epsilon, mu, sigma)
end

function MaxwellEquations3D(absorber::AbsorberModel; kwargs...)
    return MaxwellEquations3D(Homogeneous(), absorber; kwargs...)
end

function Base.similar(equations::MaxwellEquations3D{Material, Absorber, NVARS},
                      ::Type{RealT}) where {Material, Absorber, NVARS, RealT}
    return MaxwellEquations3D{Material, Absorber, NVARS}(convert(RealT, equations.epsilon),
                                                         convert(RealT, equations.mu),
                                                         convert(RealT, equations.sigma))
end

"""
    permittivity(u, equations), permeability(u, equations), conductivity(u, equations)
    impedance(u, equations), admittance(u, equations), speed_of_light(u, equations)

Material at a node: struct fields for `Homogeneous`, read from the state for
`Heterogeneous`. The one-argument forms exist for `Homogeneous` only.
"""
permittivity(equations::MaxwellEquations3D{Homogeneous}) = equations.epsilon
permeability(equations::MaxwellEquations3D{Homogeneous}) = equations.mu
conductivity(equations::MaxwellEquations3D{Homogeneous}) = equations.sigma
impedance(equations::MaxwellEquations3D{Homogeneous}) = equations.impedance
admittance(equations::MaxwellEquations3D{Homogeneous}) = equations.admittance
speed_of_light(equations::MaxwellEquations3D{Homogeneous}) = equations.speed_of_light

permittivity(u, equations::MaxwellEquations3D{Homogeneous}) = permittivity(equations)
permeability(u, equations::MaxwellEquations3D{Homogeneous}) = permeability(equations)
conductivity(u, equations::MaxwellEquations3D{Homogeneous}) = conductivity(equations)
impedance(u, equations::MaxwellEquations3D{Homogeneous}) = impedance(equations)
admittance(u, equations::MaxwellEquations3D{Homogeneous}) = admittance(equations)
speed_of_light(u, equations::MaxwellEquations3D{Homogeneous}) = speed_of_light(equations)

permittivity(u, ::MaxwellEquations3D{Heterogeneous}) = u[7]
permeability(u, ::MaxwellEquations3D{Heterogeneous}) = u[8]
conductivity(u, ::MaxwellEquations3D{Heterogeneous}) = u[9]
function impedance(u, equations::MaxwellEquations3D{Heterogeneous})
    return sqrt(permeability(u, equations) / permittivity(u, equations))
end
admittance(u, equations::MaxwellEquations3D{Heterogeneous}) = inv(impedance(u, equations))
function speed_of_light(u, equations::MaxwellEquations3D{Heterogeneous})
    return inv(sqrt(permittivity(u, equations) * permeability(u, equations)))
end

@inline function impedance_admittance(u, equations::MaxwellEquations3D{Homogeneous})
    return impedance(equations), admittance(equations)
end
# Both values with a single square root in the heterogeneous case.
@inline function impedance_admittance(u, equations::MaxwellEquations3D{Heterogeneous})
    Z = impedance(u, equations)
    return Z, inv(Z)
end

material_names(::MaxwellEquations3D{Homogeneous}) = ()
material_names(::MaxwellEquations3D{Heterogeneous}) = ("epsilon", "mu", "sigma")
pml_names(::MaxwellEquations3D{<:Any, NoPML}) = ()
pml_names(::MaxwellEquations3D{<:Any, UPML}) = ("px", "py", "pz", "qx", "qy", "qz")

function Trixi.varnames(::typeof(Trixi.cons2cons), equations::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", material_names(equations)...,
            pml_names(equations)...)
end

function Trixi.varnames(::typeof(Trixi.cons2prim), equations::MaxwellEquations3D)
    return Trixi.varnames(Trixi.cons2cons, equations)
end

@inline electric_field(u) = SVector(u[1], u[2], u[3])
@inline magnetic_field(u) = SVector(u[4], u[5], u[6])
@inline material_components(u) = SVector(u[7], u[8], u[9])

@inline pml_offset(::MaxwellEquations3D{Homogeneous}) = 6
@inline pml_offset(::MaxwellEquations3D{Heterogeneous}) = 9

@inline function pml_electric(u, equations::MaxwellEquations3D{<:Any, UPML})
    o = pml_offset(equations)
    return SVector(u[o + 1], u[o + 2], u[o + 3])
end

@inline function pml_magnetic(u, equations::MaxwellEquations3D{<:Any, UPML})
    o = pml_offset(equations)
    return SVector(u[o + 4], u[o + 5], u[o + 6])
end

# Flux, source and entropy contribution of the passive components: none.
@inline function passive_flux(::MaxwellEquations3D{Material, Absorber, NVARS, RealT}) where {
                                                                                             Material,
                                                                                             Absorber,
                                                                                             NVARS,
                                                                                             RealT
                                                                                             }
    return zero(SVector{NVARS - 6, RealT})
end

@inline function passive_components(u,
                                    ::MaxwellEquations3D{Material, Absorber, NVARS}) where {
                                                                                            Material,
                                                                                            Absorber,
                                                                                            NVARS
                                                                                            }
    return SVector(ntuple(i -> u[6 + i], Val(NVARS - 6)))
end

# Exterior state of a boundary face: given fields, passive components of the interior.
@inline function assemble(E, H, u_inner, equations::MaxwellEquations3D)
    return vcat(E, H, passive_components(u_inner, equations))
end

@inline function default_materials(::MaxwellEquations3D{Homogeneous, Absorber, NVARS,
                                                        RealT}) where {Absorber, NVARS,
                                                                       RealT}
    return SVector{0, RealT}()
end

@inline function default_materials(equations::MaxwellEquations3D{Heterogeneous})
    return SVector(equations.epsilon, equations.mu, equations.sigma)
end

@inline function zero_pml(::MaxwellEquations3D{Material, NoPML, NVARS,
                                               RealT}) where {Material, NVARS, RealT}
    return SVector{0, RealT}()
end
@inline function zero_pml(::MaxwellEquations3D{Material, UPML, NVARS,
                                               RealT}) where {Material, NVARS, RealT}
    return zero(SVector{6, RealT})
end

# Initial state: given fields, default material, auxiliary fields at rest.
@inline function with_passive_defaults(fields, equations::MaxwellEquations3D)
    return vcat(fields, default_materials(equations), zero_pml(equations))
end

@inline function Trixi.flux(u, normal_direction::AbstractVector,
                            equations::MaxwellEquations3D)
    E = electric_field(u)
    H = magnetic_field(u)
    eps = permittivity(u, equations)
    mu = permeability(u, equations)

    f_E = -cross(normal_direction, H) / eps
    f_H = cross(normal_direction, E) / mu

    return vcat(f_E, f_H, passive_flux(equations))
end

@inline function unit_normal(orientation::Integer, ::Type{RealT}) where {RealT}
    if orientation == 1
        return SVector(one(RealT), zero(RealT), zero(RealT))
    elseif orientation == 2
        return SVector(zero(RealT), one(RealT), zero(RealT))
    else
        return SVector(zero(RealT), zero(RealT), one(RealT))
    end
end

@inline function Trixi.flux(u, orientation::Integer, equations::MaxwellEquations3D)
    return Trixi.flux(u, unit_normal(orientation, eltype(u)), equations)
end

@doc raw"""
    FluxUpwindPenalty(alpha)
    flux_upwind = FluxUpwindPenalty(1.0)

Numerical flux with upwind parameter ``\alpha``: ``\alpha = 1`` is the upwind
flux, ``\alpha = 0`` the central flux. For the side with state `u_ll`, normal
``n`` and jumps ``\Delta = u_{rr} - u_{ll}``,
```math
f^*_E = -\frac{1}{\epsilon^-} \left( n \times H^- + \frac{Z^+ n \times \Delta H + \alpha |n| \Delta E_t}{Z^- + Z^+} \right), \quad
f^*_H = \frac{1}{\mu^-} \left( n \times E^- + \frac{Y^+ n \times \Delta E - \alpha |n| \Delta H_t}{Y^- + Y^+} \right),
```
with impedances ``Z``, admittances ``Y = 1/Z`` and tangential jumps
``\Delta E_t``, ``\Delta H_t``. Each side scales with its own material, so the
flux is evaluated per side across material interfaces.
- Jan S. Hesthaven, Tim Warburton (2008)
  Nodal Discontinuous Galerkin Methods, Sections 6.5 and 10.5
  [DOI: 10.1007/978-0-387-72067-8](https://doi.org/10.1007/978-0-387-72067-8)
- Kurt Busch, Michael König, Jens Niegemann (2011)
  Discontinuous Galerkin methods in nanophotonics, Eq. (6)
  [DOI: 10.1002/lpor.201000045](https://doi.org/10.1002/lpor.201000045)
"""
struct FluxUpwindPenalty{RealT <: Real}
    alpha::RealT
end

const flux_upwind = FluxUpwindPenalty(1.0)

@inline function (numerical_flux::FluxUpwindPenalty)(u_ll, u_rr,
                                                     normal_direction::AbstractVector,
                                                     equations::MaxwellEquations3D)
    RealT = eltype(u_ll)
    alpha = convert(RealT, numerical_flux.alpha)

    E_ll = electric_field(u_ll)
    H_ll = magnetic_field(u_ll)
    E_rr = electric_field(u_rr)
    H_rr = magnetic_field(u_rr)

    eps_ll = permittivity(u_ll, equations)
    mu_ll = permeability(u_ll, equations)

    Z_ll, Y_ll = impedance_admittance(u_ll, equations)

    Z_rr, Y_rr = impedance_admittance(u_rr, equations)

    norm_ = norm(normal_direction)
    n_hat = normal_direction / norm_

    dE = E_rr - E_ll
    dH = H_rr - H_ll
    dE_t = dE - dot(dE, n_hat) * n_hat
    dH_t = dH - dot(dH, n_hat) * n_hat

    f_E = -(cross(normal_direction, H_ll) +
            (Z_rr * cross(normal_direction, dH) + alpha * norm_ * dE_t) / (Z_ll + Z_rr)) /
          eps_ll
    f_H = (cross(normal_direction, E_ll) +
           (Y_rr * cross(normal_direction, dE) - alpha * norm_ * dH_t) / (Y_ll + Y_rr)) /
          mu_ll

    return vcat(f_E, f_H, passive_flux(equations))
end

@inline function (numerical_flux::FluxUpwindPenalty)(u_ll, u_rr, orientation::Integer,
                                                     equations::MaxwellEquations3D)
    return numerical_flux(u_ll, u_rr, unit_normal(orientation, eltype(u_ll)), equations)
end

function Base.show(io::IO, numerical_flux::FluxUpwindPenalty)
    print(io, "FluxUpwindPenalty(alpha=", numerical_flux.alpha, ")")
end

@doc raw"""
    initial_condition_convergence_test(x, t, equations::MaxwellEquations3D{Homogeneous})

Plane wave travelling in the positive x direction with unit period,
``E_y = \sin(2\pi (x - c t))``, ``H_z = Y E_y``.
"""
function Trixi.initial_condition_convergence_test(x, t,
                                                  equations::MaxwellEquations3D{Homogeneous})
    c = speed_of_light(equations)
    Y = admittance(equations)
    g = sinpi(2 * (x[1] - c * t))
    z = zero(g)

    return with_passive_defaults(SVector(z, g, z, z, z, g * Y), equations)
end

@doc raw"""
    initial_condition_cavity(x, t, equations::MaxwellEquations3D{Homogeneous})

Lowest TM mode of a perfectly conducting cube cavity filled with a homogeneous,
possibly lossy medium,
```math
E_z = e^{-\gamma t} \left( \cos \omega t - \frac{\gamma}{\omega} \sin \omega t \right) \sin(\pi x) \sin(\pi y),
\qquad \omega_0 = \sqrt{2} \pi c, \quad \gamma = \frac{\sigma}{2 \epsilon}, \quad \omega = \sqrt{\omega_0^2 - \gamma^2},
```
with ``H`` from Faraday's law and ``H(0) = 0``. Requires ``\gamma < \omega_0``; for
``\sigma = 0`` it is the undamped mode of B 10.5. Valid on any box whose faces
lie on integer coordinates, such as ``[-1, 1]^3`` or ``[0, 1]^3``, with
[`boundary_condition_perfect_electric_conductor`](@ref) on all faces and, for
``\sigma > 0``, [`source_terms_conductivity`](@ref).
- Jan S. Hesthaven, Tim Warburton (2008)
  Nodal Discontinuous Galerkin Methods, Section 10.5
  [DOI: 10.1007/978-0-387-72067-8](https://doi.org/10.1007/978-0-387-72067-8)
"""
function initial_condition_cavity(x, t, equations::MaxwellEquations3D{Homogeneous})
    RealT = eltype(x)
    eps = permittivity(equations)
    mu = permeability(equations)
    sigma = conductivity(equations)

    omega0 = convert(RealT, sqrt(2) * pi) * speed_of_light(equations)
    gamma = sigma / (2 * eps)
    omega = sqrt(omega0^2 - gamma^2)
    decay = exp(-gamma * t)
    amplitude = convert(RealT, pi) / (mu * omega)

    Ez = decay * (cos(omega * t) - gamma / omega * sin(omega * t)) * sinpi(x[1]) *
         sinpi(x[2])
    Hx = -amplitude * decay * sinpi(x[1]) * cospi(x[2]) * sin(omega * t)
    Hy = amplitude * decay * cospi(x[1]) * sinpi(x[2]) * sin(omega * t)
    z = zero(Ez)

    return with_passive_defaults(SVector(z, z, Ez, Hx, Hy, z), equations)
end

@inline Trixi.cons2prim(u, ::MaxwellEquations3D) = u
@inline function Trixi.cons2entropy(u, equations::MaxwellEquations3D)
    eps = permittivity(u, equations)
    mu = permeability(u, equations)
    return vcat(eps * electric_field(u), mu * magnetic_field(u), passive_flux(equations))
end

@doc raw"""
    source_terms_conductivity(u, x, t, equations::MaxwellEquations3D)

Ohmic loss ``\partial_t E = -\sigma E / \epsilon`` with the conductivity of the
material at the node.
"""
@inline function source_terms_conductivity(u, x, t, equations::MaxwellEquations3D)
    rate = -conductivity(u, equations) / permittivity(u, equations)
    return vcat(rate * electric_field(u), zero(SVector{3, eltype(u)}),
                passive_flux(equations))
end

function Trixi.energy_total(u, equations::MaxwellEquations3D)
    E = electric_field(u)
    H = magnetic_field(u)

    return 0.5f0 *
           (permittivity(u, equations) * dot(E, E) + permeability(u, equations) * dot(H, H))
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           equations::MaxwellEquations3D)
    return max(speed_of_light(u_ll, equations), speed_of_light(u_rr, equations))
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                           equations::MaxwellEquations3D)
    return Trixi.max_abs_speed_naive(u_ll, u_rr, 1, equations) * norm(normal_direction)
end

@inline function Trixi.max_abs_speeds(u, equations::MaxwellEquations3D)
    c = speed_of_light(u, equations)
    return c, c, c
end

@inline function Trixi.max_abs_speeds(equations::MaxwellEquations3D{Homogeneous})
    c = speed_of_light(equations)
    return c, c, c
end

@inline Trixi.have_constant_speed(::MaxwellEquations3D{Homogeneous}) = Trixi.True()

struct BoundaryConditionPerfectElectricConductor end
"""
    boundary_condition_perfect_electric_conductor = BoundaryConditionPerfectElectricConductor()

Perfect electric conductor: exterior state `(-E, H)` with the interior material,
evaluated with the surface flux of the scheme.
"""
const boundary_condition_perfect_electric_conductor = BoundaryConditionPerfectElectricConductor()

@inline function (::BoundaryConditionPerfectElectricConductor)(u_inner,
                                                               normal_direction::AbstractVector,
                                                               x, t, surface_flux,
                                                               equations::MaxwellEquations3D)
    u_outer = assemble(-electric_field(u_inner), magnetic_field(u_inner), u_inner,
                       equations)
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end

struct BoundaryConditionPerfectMagneticConductor end
"""
    boundary_condition_perfect_magnetic_conductor = BoundaryConditionPerfectMagneticConductor()

Perfect magnetic conductor: exterior state `(E, -H)` with the interior material,
evaluated with the surface flux of the scheme.
"""
const boundary_condition_perfect_magnetic_conductor = BoundaryConditionPerfectMagneticConductor()

@inline function (::BoundaryConditionPerfectMagneticConductor)(u_inner,
                                                               normal_direction::AbstractVector,
                                                               x, t, surface_flux,
                                                               equations::MaxwellEquations3D)
    u_outer = assemble(electric_field(u_inner), -magnetic_field(u_inner), u_inner,
                       equations)
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end

struct BoundaryConditionSilverMueller end
"""
    boundary_condition_silver_mueller = BoundaryConditionSilverMueller()

First-order absorbing boundary: [`flux_upwind`](@ref) against an exterior state
with zero fields and the interior material, independent of the surface flux of
the scheme. Exact for normal incidence only.
"""
const boundary_condition_silver_mueller = BoundaryConditionSilverMueller()

@inline function (::BoundaryConditionSilverMueller)(u_inner,
                                                    normal_direction::AbstractVector,
                                                    x, t, surface_flux,
                                                    equations::MaxwellEquations3D)
    zero_field = zero(electric_field(u_inner))
    u_outer = assemble(zero_field, zero_field, u_inner, equations)
    return flux_upwind(u_inner, u_outer, normal_direction, equations)
end

"""
    BoundaryConditionIncidentField(incident_field)

Exterior fields `incident_field(x, t, equations)` on a boundary face, combined
with the interior material. With [`flux_upwind`](@ref) only the incoming
characteristic of the incident field enters the domain.
"""
struct BoundaryConditionIncidentField{F}
    incident_field::F
end

@inline function (boundary_condition::BoundaryConditionIncidentField)(u_inner,
                                                                      normal_direction::AbstractVector,
                                                                      x, t, surface_flux,
                                                                      equations::MaxwellEquations3D)
    u_incident = boundary_condition.incident_field(x, t, equations)
    u_outer = assemble(electric_field(u_incident), magnetic_field(u_incident), u_inner,
                       equations)
    return surface_flux(u_inner, u_outer, normal_direction, equations)
end

const MaxwellBoundaryCondition = Union{BoundaryConditionPerfectElectricConductor,
                                       BoundaryConditionPerfectMagneticConductor,
                                       BoundaryConditionSilverMueller,
                                       BoundaryConditionIncidentField}

# TreeMesh passes the axis and the side instead of a normal vector.
@inline function (boundary_condition::MaxwellBoundaryCondition)(u_inner,
                                                                orientation::Integer,
                                                                direction, x, t,
                                                                surface_flux,
                                                                equations::MaxwellEquations3D)
    RealT = eltype(u_inner)
    normal_direction = SVector(ntuple(i -> i == orientation ? one(RealT) : zero(RealT),
                                      Val(3)))
    return boundary_condition(u_inner, normal_direction, direction, x, t, surface_flux,
                              equations)
end

# StructuredMesh passes inward normals on the negative sides (odd directions).
@inline function (boundary_condition::MaxwellBoundaryCondition)(u_inner,
                                                                normal_direction::AbstractVector,
                                                                direction, x, t,
                                                                surface_flux,
                                                                equations::MaxwellEquations3D)
    if isodd(direction)
        return -boundary_condition(u_inner, -normal_direction, x, t, surface_flux,
                                   equations)
    else
        return boundary_condition(u_inner, normal_direction, x, t, surface_flux,
                                  equations)
    end
end
