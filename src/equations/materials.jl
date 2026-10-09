@doc raw"""
    Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon),
             drude = (), lorentz = ())

Relative permittivity, relative permeability and normalized conductivity of a
linear medium, assigned per element with [`set_materials!`](@ref). A dispersive
medium adds tuples of [`DrudePole`](@ref)s and [`LorentzPole`](@ref)s, with
`epsilon` the high-frequency permittivity ``\epsilon_\infty``; it needs
equations with at least as many poles of each kind, the remaining poles of the
element are inactive.
"""
struct Material{RealT <: Real, Drude <: Tuple, Lorentz <: Tuple}
    epsilon::RealT
    mu::RealT
    sigma::RealT
    drude::Drude
    lorentz::Lorentz
end

Material(epsilon::Real, mu::Real, sigma::Real) = Material(epsilon, mu, sigma, (), ())

function Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon), drude = (),
                  lorentz = ())
    epsilon, mu, sigma = promote(float(epsilon), float(mu), float(sigma))
    RealT = typeof(epsilon)
    return Material(epsilon, mu, sigma,
                    map(pole -> convert(DrudePole{RealT}, pole), Tuple(drude)),
                    map(pole -> convert(LorentzPole{RealT}, pole), Tuple(lorentz)))
end

@inline material_components(material::Material) = SVector(material.epsilon, material.mu,
                                                          material.sigma)

@doc raw"""
    relative_permittivity(material::Material, frequency)

Complex relative permittivity ``\epsilon(\omega)`` of `material` at the
`frequency` ``f = \omega / 2\pi``, with the time dependence ``e^{-i \omega t}``
of [`mie_efficiencies`](@ref) and [`slab_transmittance_reflectance`](@ref), so
that absorption gives ``\operatorname{Im} \epsilon > 0``. The conductivity
contributes ``i \sigma / \omega``.
"""
function relative_permittivity(material::Material, frequency)
    omega = 2 * pi * frequency
    epsilon = complex(material.epsilon) + im * material.sigma / omega
    for pole in material.drude
        epsilon -= pole.plasma_frequency^2 / (omega^2 + im * pole.damping * omega)
    end
    for pole in material.lorentz
        epsilon += pole.delta_epsilon * pole.resonance_frequency^2 /
                   (pole.resonance_frequency^2 - omega^2 - im * pole.damping * omega)
    end
    return epsilon
end

# Passive components of the state for `material`: epsilon, mu, sigma and the pole
# parameters, padded with inactive poles up to the poles of the equations.
function material_state(material::Material,
                        equations::MaxwellEquations3D{Heterogeneous, Dispersion}) where {Dispersion}
    num_drude = num_drude_poles(Dispersion)
    num_lorentz = num_lorentz_poles(Dispersion)
    if length(material.drude) > num_drude || length(material.lorentz) > num_lorentz
        throw(ArgumentError("the material has $(length(material.drude)) Drude and $(length(material.lorentz)) Lorentz poles, the equations only $num_drude and $num_lorentz"))
    end
    RealT = eltype(equations.epsilon)
    drude = ntuple(k -> k <= length(material.drude) ? material.drude[k] :
                        DrudePole(zero(RealT), zero(RealT)), num_drude)
    lorentz = ntuple(k -> k <= length(material.lorentz) ? material.lorentz[k] :
                          LorentzPole(zero(RealT), zero(RealT), zero(RealT)), num_lorentz)
    return vcat(material_components(material),
                pole_parameter_vector(drude, lorentz, RealT))
end

function element_centroid(md, element)
    return SVector(ntuple(d -> sum(view(md.xyz[d], :, element)) / size(md.xyz[d], 1), 3))
end

function element_centroid(node_coordinates::AbstractArray{<:Any, 5}, element)
    nodes = view(node_coordinates, :, :, :, :, element)
    return SVector(ntuple(d -> sum(view(nodes, d, :, :, :)) /
                               length(view(nodes, d, :, :, :)),
                          3))
end

centroid_source(mesh::DGMultiMesh, cache) = mesh.md
centroid_source(mesh, cache) = cache.elements.node_coordinates

"""
    set_materials!(u_ode, semi, material_at)
    set_materials!(u_ode, semi, element_groups, materials::AbstractDict)

Write the material into the passive components of every node of a
[`MaxwellEquations3D`](@ref)`{Heterogeneous}` state, element by element. The
first form evaluates `material_at(x)` at the element centroid, the second looks
up `materials[element_groups[element]]`, for example with the groups of an
[`ImportedMesh`](@ref). Both return a [`Material`](@ref) per element. Call after
`semidiscretize` on `ode.u0`.
"""
function set_materials!(u_ode, semi, material_at)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    if !(equations isa MaxwellEquations3D{Heterogeneous})
        throw(ArgumentError("set_materials! needs MaxwellEquations3D(Heterogeneous()), got $(typeof(equations).name.wrapper) with material $(typeof(equations).parameters[1])"))
    end
    geometry = centroid_source(mesh, cache)
    return set_materials_by_element!(u_ode, semi,
                                     element -> material_state(material_at(element_centroid(geometry,
                                                                                            element)),
                                                               equations))
end

function set_materials!(u_ode, semi, element_groups::AbstractVector{<:Integer},
                        materials::AbstractDict)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    num_elements = Trixi.nelements(mesh, solver, cache)
    length(element_groups) == num_elements ||
        throw(ArgumentError("got $(length(element_groups)) element groups for $num_elements elements"))
    for group in unique(element_groups)
        haskey(materials, group) ||
            throw(ArgumentError("no material given for element group $group"))
    end
    return set_materials_by_element!(u_ode, semi,
                                     element -> material_state(materials[element_groups[element]],
                                                               equations))
end

function set_materials_by_element!(u_ode, semi, material_of_element)
    u = Trixi.wrap_array(u_ode, semi)
    return set_materials_by_element!(u, material_of_element)
end

# `state_of_element(element)` returns the passive components from index 7 on.
function set_materials_by_element!(u::AbstractArray{<:Any, 5}, state_of_element)
    for element in axes(u, 5)
        components = state_of_element(element)
        for k in axes(u, 4), j in axes(u, 3), i in axes(u, 2), v in eachindex(components)
            u[6 + v, i, j, k, element] = components[v]
        end
    end
    return u
end

function set_materials_by_element!(u, state_of_element)
    for element in axes(u, 2)
        components = state_of_element(element)
        for node in axes(u, 1)
            u_node = u[node, element]
            for v in eachindex(components)
                u_node = Base.setindex(u_node, components[v], 6 + v)
            end
            u[node, element] = u_node
        end
    end
    return u
end
