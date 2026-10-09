@doc raw"""
    PMLProfile(coordinates_min, coordinates_max, thickness;
               order = 3, reflection = 1e-6, impedance = 1.0)

Damping profile of a uniaxial perfectly matched layer filling the shell of the
given `thickness` inside the box `coordinates_min`, `coordinates_max`, the
same values as passed to `DGMultiMesh`. Along each axis
``\sigma_i(x) = \sigma_\mathrm{max} (d_i / \mathrm{thickness})^\mathrm{order}``
with the depth ``d_i`` into the layer, zero in the physical region. The default
``\sigma_\mathrm{max} = (\mathrm{order} + 1) \ln(1 / \mathrm{reflection}) /
(2 Z \, \mathrm{thickness})`` targets the given normal-incidence reflection of
the layer terminated by [`boundary_condition_silver_mueller`](@ref); pass
`sigma_max` to override it.
"""
struct PMLProfile{RealT <: Real}
    coordinates_min::SVector{3, RealT}
    coordinates_max::SVector{3, RealT}
    thickness::RealT
    sigma_max::RealT
    order::Int
end

function PMLProfile(coordinates_min, coordinates_max, thickness; order = 3,
                    reflection = 1.0e-6, impedance = 1.0,
                    sigma_max = (order + 1) * log(1 / reflection) /
                                (2 * impedance * thickness))
    xmin, xmax = promote(SVector{3}(coordinates_min), SVector{3}(coordinates_max))
    RealT = eltype(xmin)
    return PMLProfile(xmin, xmax, convert(RealT, thickness), convert(RealT, sigma_max),
                      order)
end

@inline function (profile::PMLProfile)(x)
    (; coordinates_min, coordinates_max, thickness, sigma_max, order) = profile
    depth = max.(x .- (coordinates_max .- thickness), (coordinates_min .+ thickness) .- x,
                 zero(thickness))
    return sigma_max .* (depth ./ thickness) .^ order
end

@doc raw"""
    SourceTermsPML(profile)

Source term of the uniaxial perfectly matched layer with auxiliary differential
equations (Busch, König, Niegemann 2011, eq. 38). With the damping
``\sigma = (\sigma_x, \sigma_y, \sigma_z)`` of the `profile` at the node and
cyclic indices,
```math
\partial_t E_x = \ldots - (\sigma_y + \sigma_z - \sigma_x) E_x - p_x / \epsilon, \qquad
\partial_t p_x = (\sigma_x - \sigma_y)(\sigma_x - \sigma_z) \epsilon E_x - \sigma_x p_x,
```
and the same for ``H`` and ``q`` with ``\mu``. Requires
[`MaxwellEquations3D`](@ref) with `UPML()`; the layer must continue the
background material of the adjacent physical region.
"""
struct SourceTermsPML{Profile}
    profile::Profile
end

@inline function (source::SourceTermsPML)(u, x, t,
                                          equations::MaxwellEquations3D{<:Any, <:Any, UPML})
    sigma = source.profile(x)
    E = electric_field(u)
    H = magnetic_field(u)
    p = pml_electric(u, equations)
    q = pml_magnetic(u, equations)
    eps = permittivity(u, equations)
    mu = permeability(u, equations)

    damping = SVector(sigma[2] + sigma[3] - sigma[1],
                      sigma[3] + sigma[1] - sigma[2],
                      sigma[1] + sigma[2] - sigma[3])
    coupling = SVector((sigma[1] - sigma[2]) * (sigma[1] - sigma[3]),
                       (sigma[2] - sigma[3]) * (sigma[2] - sigma[1]),
                       (sigma[3] - sigma[1]) * (sigma[3] - sigma[2]))

    dE = -damping .* E - p / eps
    dH = -damping .* H - q / mu
    dp = coupling .* (eps * E) - sigma .* p
    dq = coupling .* (mu * H) - sigma .* q
    return vcat(dE, dH, zero_before_pml(equations), dp, dq)
end

function (::SourceTermsPML)(u, x, t, equations::MaxwellEquations3D)
    throw(ArgumentError("SourceTermsPML needs MaxwellEquations3D with UPML()"))
end

"""
    CombinedSourceTerms(terms...)

Sum of several source terms `term(u, x, t, equations)`, for example
[`SourceTermsPML`](@ref) together with a [`HertzianDipole`](@ref).
"""
struct CombinedSourceTerms{Terms <: Tuple}
    terms::Terms
end

CombinedSourceTerms(terms...) = CombinedSourceTerms(terms)

@inline function (source::CombinedSourceTerms)(u, x, t, equations)
    return mapreduce(term -> term(u, x, t, equations), +, source.terms)
end
