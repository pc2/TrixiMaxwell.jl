struct Homogeneous end
struct Heterogeneous end
const MaterialModel = Union{Homogeneous, Heterogeneous}

struct NoPML end
struct UPML end
const AbsorberModel = Union{NoPML, UPML}

@doc raw"""
    DrudePole(plasma_frequency, damping)

Drude contribution ``-\omega_p^2 / (\omega^2 + i \gamma \omega)`` to the relative
permittivity, with the plasma frequency ``\omega_p`` and the damping ``\gamma`` as
angular frequencies in the normalized units of [`MaxwellEquations3D`](@ref).
A pole with zero plasma frequency is inactive.
"""
struct DrudePole{RealT <: Real}
    plasma_frequency::RealT
    damping::RealT
end

DrudePole(plasma_frequency, damping) = DrudePole(promote(plasma_frequency, damping)...)

@doc raw"""
    LorentzPole(delta_epsilon, resonance_frequency, damping)

Lorentz contribution
``\Delta\epsilon \, \omega_L^2 / (\omega_L^2 - \omega^2 - i \delta \omega)`` to the
relative permittivity, with the strength ``\Delta\epsilon``, the resonance
frequency ``\omega_L`` and the damping ``\delta`` as angular frequencies. A pole
with zero strength is inactive.
"""
struct LorentzPole{RealT <: Real}
    delta_epsilon::RealT
    resonance_frequency::RealT
    damping::RealT
end

function LorentzPole(delta_epsilon, resonance_frequency, damping)
    return LorentzPole(promote(delta_epsilon, resonance_frequency, damping)...)
end

function Base.convert(::Type{DrudePole{RealT}}, pole::DrudePole) where {RealT}
    DrudePole{RealT}(pole.plasma_frequency, pole.damping)
end
function Base.convert(::Type{LorentzPole{RealT}}, pole::LorentzPole) where {RealT}
    return LorentzPole{RealT}(pole.delta_epsilon, pole.resonance_frequency, pole.damping)
end

struct NonDispersive end

"""
    Dispersive(drude, lorentz)

Dispersion model of [`MaxwellEquations3D`](@ref): tuples of [`DrudePole`](@ref)s
and [`LorentzPole`](@ref)s. Built by the equations constructor from its `drude`
and `lorentz` keywords.
"""
struct Dispersive{NDrude, NLorentz, RealT <: Real}
    drude::NTuple{NDrude, DrudePole{RealT}}
    lorentz::NTuple{NLorentz, LorentzPole{RealT}}
end

const DispersionModel = Union{NonDispersive, Dispersive}

function dispersion_model(drude::Tuple, lorentz::Tuple, ::Type{RealT}) where {RealT}
    isempty(drude) && isempty(lorentz) && return NonDispersive()
    return Dispersive(map(pole -> convert(DrudePole{RealT}, pole), drude),
                      map(pole -> convert(LorentzPole{RealT}, pole), lorentz))
end

convert_dispersion(::Type{RealT}, dispersion::NonDispersive) where {RealT} = dispersion
function convert_dispersion(::Type{RealT}, dispersion::Dispersive) where {RealT}
    return dispersion_model(dispersion.drude, dispersion.lorentz, RealT)
end

@doc raw"""
    MaxwellEquations3D(material = Homogeneous(), absorber = NoPML();
                       epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon),
                       drude = (), lorentz = ())
    MaxwellEquations3D(UPML(); kwargs...)

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

Dispersive media follow from the `drude` and `lorentz` tuples of
[`DrudePole`](@ref)s and [`LorentzPole`](@ref)s; `epsilon` is then the
high-frequency permittivity ``\epsilon_\infty``. Each Drude pole adds a
polarization current `(Jx, Jy, Jz)`, each Lorentz pole a polarization and its
current `(Px, Py, Pz, Jx, Jy, Jz)`, all with zero flux and zero initial value,
driven by [`source_terms_dispersive`](@ref). With `Homogeneous` the poles
describe the medium; with `Heterogeneous` the state also carries the pole
parameters per node, set with [`set_materials!`](@ref), and the poles of the
struct are their defaults.
"""
struct MaxwellEquations3D{Material, Dispersion, Absorber, NVARS, RealT <: Real} <:
       Trixi.AbstractMaxwellEquations{3, NVARS}
    epsilon::RealT
    mu::RealT
    sigma::RealT
    impedance::RealT
    admittance::RealT
    speed_of_light::RealT
    dispersion::Dispersion

    function MaxwellEquations3D{Material, Absorber}(epsilon, mu, sigma,
                                                    dispersion) where {Material, Absorber}
        impedance = sqrt(mu / epsilon)
        epsilon, mu, sigma, impedance, admittance, speed_of_light = promote(epsilon, mu,
                                                                            sigma,
                                                                            impedance,
                                                                            inv(impedance),
                                                                            inv(sqrt(epsilon *
                                                                                     mu)))
        RealT = typeof(epsilon)
        dispersion = convert_dispersion(RealT, dispersion)
        Dispersion = typeof(dispersion)
        NVARS = 6 + num_material_components(Material) +
                num_pole_parameters(Material, Dispersion) +
                num_current_components(Dispersion) + num_pml_components(Absorber)
        return new{Material, Dispersion, Absorber, NVARS, RealT}(epsilon, mu, sigma,
                                                                 impedance, admittance,
                                                                 speed_of_light,
                                                                 dispersion)
    end
end

num_material_components(::Type{Homogeneous}) = 0
num_material_components(::Type{Heterogeneous}) = 3

num_drude_poles(::Type{NonDispersive}) = 0
num_drude_poles(::Type{<:Dispersive{NDrude}}) where {NDrude} = NDrude
num_lorentz_poles(::Type{NonDispersive}) = 0
function num_lorentz_poles(::Type{<:Dispersive{NDrude, NLorentz}}) where {NDrude, NLorentz}
    NLorentz
end

num_pole_parameters(::Type{Homogeneous}, Dispersion) = 0
function num_pole_parameters(::Type{Heterogeneous}, Dispersion)
    return 2 * num_drude_poles(Dispersion) + 3 * num_lorentz_poles(Dispersion)
end
function num_current_components(Dispersion)
    return 3 * num_drude_poles(Dispersion) + 6 * num_lorentz_poles(Dispersion)
end

num_pml_components(::Type{NoPML}) = 0
num_pml_components(::Type{UPML}) = 6

function MaxwellEquations3D(material::Material = Homogeneous(),
                            absorber::Absorber = NoPML();
                            epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon),
                            drude = (),
                            lorentz = ()) where {Material <: MaterialModel,
                                                 Absorber <: AbsorberModel}
    RealT = float(promote_type(typeof(epsilon), typeof(mu), typeof(sigma)))
    return MaxwellEquations3D{Material, Absorber}(epsilon, mu, sigma,
                                                  dispersion_model(Tuple(drude),
                                                                   Tuple(lorentz), RealT))
end

function MaxwellEquations3D(absorber::AbsorberModel; kwargs...)
    return MaxwellEquations3D(Homogeneous(), absorber; kwargs...)
end

function Base.similar(equations::MaxwellEquations3D{Material, Dispersion, Absorber},
                      ::Type{RealT}) where {Material, Dispersion, Absorber, RealT}
    return MaxwellEquations3D{Material, Absorber}(convert(RealT, equations.epsilon),
                                                  convert(RealT, equations.mu),
                                                  convert(RealT, equations.sigma),
                                                  convert_dispersion(RealT,
                                                                     equations.dispersion))
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
pml_names(::MaxwellEquations3D{<:Any, <:Any, NoPML}) = ()
pml_names(::MaxwellEquations3D{<:Any, <:Any, UPML}) = ("px", "py", "pz", "qx", "qy", "qz")

function pole_parameter_names(equations::MaxwellEquations3D{Material,
                                                            Dispersion}) where {Material,
                                                                                Dispersion}
    num_pole_parameters(Material, Dispersion) == 0 && return ()
    drude = ((("omega_p_d$k", "gamma_d$k") for k in 1:num_drude_poles(Dispersion))...,)
    lorentz = ((("delta_epsilon_l$k", "omega_l$k", "delta_l$k")
                for k in 1:num_lorentz_poles(Dispersion))...,)
    return (Iterators.flatten((drude..., lorentz...))...,)
end

function current_names(::MaxwellEquations3D{Material, Dispersion}) where {Material,
                                                                          Dispersion}
    drude = ((("Jx_d$k", "Jy_d$k", "Jz_d$k") for k in 1:num_drude_poles(Dispersion))...,)
    lorentz = ((("Px_l$k", "Py_l$k", "Pz_l$k", "Jx_l$k", "Jy_l$k", "Jz_l$k")
                for k in 1:num_lorentz_poles(Dispersion))...,)
    return (Iterators.flatten((drude..., lorentz...))...,)
end

function Trixi.varnames(::typeof(Trixi.cons2cons), equations::MaxwellEquations3D)
    return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", material_names(equations)...,
            pole_parameter_names(equations)..., current_names(equations)...,
            pml_names(equations)...)
end

function Trixi.varnames(::typeof(Trixi.cons2prim), equations::MaxwellEquations3D)
    return Trixi.varnames(Trixi.cons2cons, equations)
end

@inline electric_field(u) = SVector(u[1], u[2], u[3])
@inline magnetic_field(u) = SVector(u[4], u[5], u[6])
@inline material_components(u) = SVector(u[7], u[8], u[9])

# State layout: fields, materials, pole parameters, currents, PML fields.
@inline function pole_parameter_offset(::MaxwellEquations3D{Material}) where {Material}
    return 6 + num_material_components(Material)
end
@inline function current_offset(equations::MaxwellEquations3D{Material,
                                                              Dispersion}) where {Material,
                                                                                  Dispersion
                                                                                  }
    return pole_parameter_offset(equations) + num_pole_parameters(Material, Dispersion)
end
@inline function pml_offset(equations::MaxwellEquations3D{Material,
                                                          Dispersion}) where {Material,
                                                                              Dispersion}
    return current_offset(equations) + num_current_components(Dispersion)
end

"""
    drude_pole(u, equations, k), lorentz_pole(u, equations, k)

Pole `k` at a node: from the struct for `Homogeneous`, from the state for
`Heterogeneous`.
"""
@inline drude_pole(u, equations::MaxwellEquations3D{Homogeneous, <:Dispersive}, k) = equations.dispersion.drude[k]
@inline function lorentz_pole(u, equations::MaxwellEquations3D{Homogeneous, <:Dispersive},
                              k)
    return equations.dispersion.lorentz[k]
end
@inline function drude_pole(u, equations::MaxwellEquations3D{Heterogeneous, <:Dispersive},
                            k)
    o = pole_parameter_offset(equations) + 2 * (k - 1)
    return DrudePole(u[o + 1], u[o + 2])
end
@inline function lorentz_pole(u,
                              equations::MaxwellEquations3D{Heterogeneous,
                                                            Dispersion},
                              k) where {Dispersion <: Dispersive}
    o = pole_parameter_offset(equations) + 2 * num_drude_poles(Dispersion) + 3 * (k - 1)
    return LorentzPole(u[o + 1], u[o + 2], u[o + 3])
end

@inline function drude_current(u, equations::MaxwellEquations3D, k)
    o = current_offset(equations) + 3 * (k - 1)
    return SVector(u[o + 1], u[o + 2], u[o + 3])
end
@inline function lorentz_polarization(u,
                                      equations::MaxwellEquations3D{Material,
                                                                    Dispersion},
                                      k) where {Material, Dispersion}
    o = current_offset(equations) + 3 * num_drude_poles(Dispersion) + 6 * (k - 1)
    return SVector(u[o + 1], u[o + 2], u[o + 3])
end
@inline function lorentz_current(u,
                                 equations::MaxwellEquations3D{Material, Dispersion},
                                 k) where {Material, Dispersion}
    o = current_offset(equations) + 3 * num_drude_poles(Dispersion) + 6 * (k - 1) + 3
    return SVector(u[o + 1], u[o + 2], u[o + 3])
end

@inline function pml_electric(u, equations::MaxwellEquations3D{<:Any, <:Any, UPML})
    o = pml_offset(equations)
    return SVector(u[o + 1], u[o + 2], u[o + 3])
end

@inline function pml_magnetic(u, equations::MaxwellEquations3D{<:Any, <:Any, UPML})
    o = pml_offset(equations)
    return SVector(u[o + 4], u[o + 5], u[o + 6])
end

# Flux, source and entropy contribution of the passive components: none.
@inline function passive_flux(::MaxwellEquations3D{Material, Dispersion, Absorber, NVARS,
                                                   RealT}) where {Material, Dispersion,
                                                                  Absorber, NVARS, RealT}
    return zero(SVector{NVARS - 6, RealT})
end

@inline function passive_components(u,
                                    ::MaxwellEquations3D{Material, Dispersion, Absorber,
                                                         NVARS}) where {Material,
                                                                        Dispersion,
                                                                        Absorber, NVARS}
    return SVector(ntuple(i -> u[6 + i], Val(NVARS - 6)))
end

# Zeros for the materials, pole parameters and currents, the components between
# the fields and the PML fields.
@inline function zero_before_pml(::MaxwellEquations3D{Material, Dispersion, Absorber,
                                                      NVARS, RealT}) where {Material,
                                                                            Dispersion,
                                                                            Absorber, NVARS,
                                                                            RealT}
    return zero(SVector{NVARS - 6 - num_pml_components(Absorber), RealT})
end

# Exterior state of a boundary face: given fields, passive components of the interior.
@inline function assemble(E, H, u_inner, equations::MaxwellEquations3D)
    return vcat(E, H, passive_components(u_inner, equations))
end

@inline function default_materials(::MaxwellEquations3D{Homogeneous, Dispersion,
                                                        Absorber, NVARS,
                                                        RealT}) where {Dispersion, Absorber,
                                                                       NVARS, RealT}
    return SVector{0, RealT}()
end

@inline function default_materials(equations::MaxwellEquations3D{Heterogeneous})
    return SVector(equations.epsilon, equations.mu, equations.sigma)
end

@inline pole_parameters(pole::DrudePole) = SVector(pole.plasma_frequency, pole.damping)
@inline function pole_parameters(pole::LorentzPole)
    return SVector(pole.delta_epsilon, pole.resonance_frequency, pole.damping)
end

# Pole parameters of the state, in the order of the layout.
@inline function pole_parameter_vector(drude::Tuple, lorentz::Tuple,
                                       ::Type{RealT}) where {RealT}
    return vcat(SVector{0, RealT}(), map(pole_parameters, drude)...,
                map(pole_parameters, lorentz)...)
end

@inline function default_pole_parameters(equations::MaxwellEquations3D{Material, Dispersion,
                                                                       Absorber, NVARS,
                                                                       RealT}) where {
                                                                                      Material,
                                                                                      Dispersion,
                                                                                      Absorber,
                                                                                      NVARS,
                                                                                      RealT
                                                                                      }
    return SVector{0, RealT}()
end
@inline function default_pole_parameters(equations::MaxwellEquations3D{Heterogeneous,
                                                                       <:Dispersive})
    (; drude, lorentz) = equations.dispersion
    return pole_parameter_vector(drude, lorentz, eltype(equations.epsilon))
end

@inline function zero_currents(::MaxwellEquations3D{Material, Dispersion, Absorber, NVARS,
                                                    RealT}) where {Material, Dispersion,
                                                                   Absorber, NVARS, RealT}
    return zero(SVector{num_current_components(Dispersion), RealT})
end

@inline function zero_pml(::MaxwellEquations3D{Material, Dispersion, Absorber, NVARS,
                                               RealT}) where {Material, Dispersion,
                                                              Absorber, NVARS, RealT}
    return zero(SVector{num_pml_components(Absorber), RealT})
end

# Initial state: given fields, default material and poles, auxiliary fields at rest.
@inline function with_passive_defaults(fields, equations::MaxwellEquations3D)
    return vcat(fields, default_materials(equations), default_pole_parameters(equations),
                zero_currents(equations), zero_pml(equations))
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

@doc raw"""
    source_terms_dispersive(u, x, t, equations::MaxwellEquations3D)

Polarization currents of the Drude and Lorentz poles of the medium at the node
(auxiliary differential equations, Busch, König, Niegemann 2011, eqs. 31 and 33),
```math
\epsilon_\infty \partial_t E = \ldots - \sum_k J_k, \qquad
\partial_t J_k = \omega_{p,k}^2 E - \gamma_k J_k \quad \text{(Drude)},
\qquad
\partial_t P_k = J_k, \quad
\partial_t J_k = \Delta\epsilon_k \omega_{L,k}^2 E - \omega_{L,k}^2 P_k - \delta_k J_k
\quad \text{(Lorentz)},
```
which give the permittivity of [`DrudePole`](@ref) and [`LorentzPole`](@ref).
Combine with other sources through [`CombinedSourceTerms`](@ref).
- Kurt Busch, Michael König, Jens Niegemann (2011)
  Discontinuous Galerkin methods in nanophotonics
  [DOI: 10.1002/lpor.201000045](https://doi.org/10.1002/lpor.201000045)
"""
@inline function source_terms_dispersive(u, x, t,
                                         equations::MaxwellEquations3D{Material,
                                                                       Dispersion}) where {
                                                                                           Material,
                                                                                           Dispersion <:
                                                                                           Dispersive
                                                                                           }
    E = electric_field(u)
    drude = ntuple(Val(num_drude_poles(Dispersion))) do k
        pole = drude_pole(u, equations, k)
        J = drude_current(u, equations, k)
        return J, pole.plasma_frequency^2 * E - pole.damping * J
    end
    lorentz = ntuple(Val(num_lorentz_poles(Dispersion))) do k
        pole = lorentz_pole(u, equations, k)
        P = lorentz_polarization(u, equations, k)
        J = lorentz_current(u, equations, k)
        dJ = pole.delta_epsilon * pole.resonance_frequency^2 * E -
             pole.resonance_frequency^2 * P - pole.damping * J
        return J, vcat(J, dJ)
    end
    current = zero(E)
    for (J, _) in drude
        current += J
    end
    for (J, _) in lorentz
        current += J
    end
    RealT = eltype(u)
    zero_passive = zero(SVector{current_offset(equations) - 6, RealT})
    return vcat(-current / permittivity(u, equations), zero(E), zero_passive,
                map(last, drude)..., map(last, lorentz)..., zero_pml(equations))
end

@doc raw"""
    initial_condition_cavity(x, t, equations::MaxwellEquations3D{Homogeneous, <:Dispersive})

Lowest TM mode of a perfectly conducting cube cavity filled with a lossless Drude
medium with one pole, ``\gamma = 0``, starting with ``H = J = 0``:
```math
E_z = \cos(\omega t) \sin(\pi x) \sin(\pi y), \qquad
\omega^2 = \frac{\omega_0^2 + \omega_p^2}{\epsilon_\infty}, \quad \omega_0 = \sqrt{2} \pi / \sqrt{\mu},
```
with ``H`` from Faraday's law and ``J_z = \omega_p^2 \sin(\omega t) / \omega \,
\sin(\pi x) \sin(\pi y)``. Requires ``\sigma = 0``.
"""
function initial_condition_cavity(x, t,
                                  equations::MaxwellEquations3D{Homogeneous,
                                                                <:Dispersive{1, 0}})
    RealT = eltype(x)
    eps = permittivity(equations)
    mu = permeability(equations)
    pole = only(equations.dispersion.drude)
    (iszero(pole.damping) && iszero(conductivity(equations))) ||
        throw(ArgumentError("the dispersive cavity mode needs a lossless Drude pole and sigma = 0"))

    omega = sqrt((2 * convert(RealT, pi)^2 / mu + pole.plasma_frequency^2) / eps)
    shape = sinpi(x[1]) * sinpi(x[2])
    amplitude = convert(RealT, pi) / (mu * omega)

    Ez = cos(omega * t) * shape
    Hx = -amplitude * sinpi(x[1]) * cospi(x[2]) * sin(omega * t)
    Hy = amplitude * cospi(x[1]) * sinpi(x[2]) * sin(omega * t)
    Jz = pole.plasma_frequency^2 * sin(omega * t) / omega * shape
    z = zero(Ez)

    return vcat(SVector(z, z, Ez, Hx, Hy, z), SVector(z, z, Jz), zero_pml(equations))
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
