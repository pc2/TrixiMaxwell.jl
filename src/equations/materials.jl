"""
    Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))

Relative permittivity, relative permeability and normalized conductivity of a
linear medium, assigned per element with [`set_materials!`](@ref).
"""
struct Material{RealT <: Real}
    epsilon::RealT
    mu::RealT
    sigma::RealT
end

function Material(; epsilon = 1.0, mu = one(epsilon), sigma = zero(epsilon))
    return Material(promote(float(epsilon), float(mu), float(sigma))...)
end

@inline material_components(material::Material) = SVector(material.epsilon, material.mu,
                                                          material.sigma)

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
                                     element -> material_at(element_centroid(geometry,
                                                                             element)))
end

function set_materials!(u_ode, semi, element_groups::AbstractVector{<:Integer},
                        materials::AbstractDict)
    mesh, _, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    num_elements = Trixi.nelements(mesh, solver, cache)
    length(element_groups) == num_elements ||
        throw(ArgumentError("got $(length(element_groups)) element groups for $num_elements elements"))
    for group in unique(element_groups)
        haskey(materials, group) ||
            throw(ArgumentError("no material given for element group $group"))
    end
    return set_materials_by_element!(u_ode, semi,
                                     element -> materials[element_groups[element]])
end

function set_materials_by_element!(u_ode, semi, material_of_element)
    u = Trixi.wrap_array(u_ode, semi)
    return set_materials_by_element!(u, material_of_element)
end

function set_materials_by_element!(u::AbstractArray{<:Any, 5}, material_of_element)
    for element in axes(u, 5)
        components = material_components(material_of_element(element))
        for k in axes(u, 4), j in axes(u, 3), i in axes(u, 2)
            u[7, i, j, k, element] = components[1]
            u[8, i, j, k, element] = components[2]
            u[9, i, j, k, element] = components[3]
        end
    end
    return u
end

function set_materials_by_element!(u, material_of_element)
    for element in axes(u, 2)
        components = material_components(material_of_element(element))
        for node in axes(u, 1)
            u_node = u[node, element]
            u[node, element] = Base.setindex(Base.setindex(Base.setindex(u_node,
                                                                         components[1], 7),
                                                           components[2], 8),
                                             components[3], 9)
        end
    end
    return u
end
