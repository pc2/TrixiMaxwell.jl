"""
    ProjectedSourceTerms(source_terms, equations, dg::DGSEM;
                         quadrature_degree = 2 * polydeg(dg) + 2,
                         support = source_support(source_terms))

Source terms for `DGSEM` that are integrated with a Gauss quadrature of the
given degree in every element and projected onto the nodal basis, instead of
being sampled at the Lobatto nodes. This keeps the total strength of sources
narrower than the node spacing, such as a [`HertzianDipole`](@ref). On other
solvers the wrapper is transparent.
"""
struct ProjectedSourceTerms{Source, RealT <: Real}
    source_terms::Source
    interpolation::Matrix{RealT}   # nodes -> quadrature points
    projection::Matrix{RealT}      # quadrature points -> nodes
    center::SVector{3, RealT}
    radius::RealT
    jacobian_threaded::Vector{Vector{RealT}}
    values_threaded::Vector{Matrix{RealT}}
    states_threaded::Vector{Matrix{RealT}}
end

"""
    source_support(source_terms)

Ball `(center, radius)` outside of which `source_terms` is smooth enough to be
sampled at the nodes, or `nothing` to project everywhere.
"""
source_support(source_terms) = nothing
source_support(dipole::HertzianDipole) = (dipole.position, 5 * dipole.width)
function source_support(combined::CombinedSourceTerms)
    supports = filter(!isnothing, map(source_support, combined.terms))
    length(supports) <= 1 ||
        throw(ArgumentError("more than one term of the combined sources declares a support"))
    return isempty(supports) ? nothing : only(supports)
end

function ProjectedSourceTerms(source_terms, equations, dg::DGSEM;
                              quadrature_degree = 2 * Trixi.polydeg(dg) + 2,
                              support = source_support(source_terms))
    nodes, weights = dg.basis.nodes, dg.basis.weights
    RealT = eltype(nodes)
    num_quadrature = cld(quadrature_degree + 1, 2)
    quadrature_nodes, quadrature_weights = Trixi.gauss_nodes_weights(num_quadrature,
                                                                     RealT)
    vandermonde = Trixi.polynomial_interpolation_matrix(nodes, quadrature_nodes)
    projection_1d = Trixi.Diagonal(inv.(weights)) * vandermonde' *
                    Trixi.Diagonal(quadrature_weights)
    interpolation = kron(vandermonde, vandermonde, vandermonde)
    projection = kron(projection_1d, projection_1d, projection_1d)
    num_points = size(interpolation, 1)
    jacobian_threaded = [zeros(RealT, num_points) for _ in 1:Threads.nthreads()]
    values_threaded = [zeros(RealT, Trixi.nvariables(equations), num_points)
                       for _ in 1:Threads.nthreads()]
    states_threaded = [zeros(RealT, Trixi.nvariables(equations), num_points)
                       for _ in 1:Threads.nthreads()]
    center = support === nothing ? zero(SVector{3, RealT}) : SVector{3, RealT}(support[1])
    radius = support === nothing ? convert(RealT, Inf) : convert(RealT, support[2])
    return ProjectedSourceTerms(source_terms, Matrix(interpolation), Matrix(projection),
                                center, radius, jacobian_threaded,
                                values_threaded, states_threaded)
end

# Distance from the ball around the support center to the node bounding box.
function outside_support(source::ProjectedSourceTerms, element_coordinates)
    isinf(source.radius) && return false
    distance2 = zero(source.radius)
    for d in 1:3
        lower, upper = extrema(view(element_coordinates, d, :))
        gap = max(lower - source.center[d], source.center[d] - upper, zero(lower))
        distance2 += gap^2
    end
    return distance2 > source.radius^2
end

@inline function (source::ProjectedSourceTerms)(u, x, t, equations)
    return source.source_terms(u, x, t, equations)
end

function Base.show(io::IO, source::ProjectedSourceTerms)
    print(io, "ProjectedSourceTerms(", source.source_terms, ", ",
          size(source.interpolation, 1), " quadrature points)")
end

# Jacobian at the nodes of one element; TreeMesh stores one value per element.
@inline element_jacobian(inverse_jacobian::AbstractVector, n, element) = inv(inverse_jacobian[element])
@inline function element_jacobian(inverse_jacobian::AbstractArray{<:Any, 4}, n, element)
    return inv(view(inverse_jacobian, :, :, :, element)[n])
end

function Trixi.calc_sources!(backend::Nothing, du, u, t, source::ProjectedSourceTerms,
                             equations::MaxwellEquations3D, dg::DGSEM, cache)
    (; node_coordinates, inverse_jacobian) = cache.elements
    (; interpolation, projection) = source
    num_nodes = size(projection, 1)
    num_points = size(projection, 2)

    Trixi.@threaded for element in Trixi.eachelement(dg, cache)
        jacobian = source.jacobian_threaded[Threads.threadid()]
        element_coordinates = reshape(view(node_coordinates, :, :, :, :, element), 3, :)
        if outside_support(source, element_coordinates)
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                u_node = Trixi.get_node_vars(u, equations, dg, i, j, k, element)
                x = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, k,
                                          element)
                Trixi.add_to_node_vars!(du, source.source_terms(u_node, x, t, equations),
                                        equations, dg, i, j, k, element)
            end
            continue
        end
        values = source.values_threaded[Threads.threadid()]
        states = source.states_threaded[Threads.threadid()]
        for q in Base.OneTo(num_points)
            x = zero(SVector{3, eltype(jacobian)})
            jq = zero(eltype(jacobian))
            for v in axes(states, 1)
                states[v, q] = 0
            end
            n = 0
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                n += 1
                weight = interpolation[q, n]
                x += weight * SVector{3}(view(element_coordinates, :, n))
                jq += weight * element_jacobian(inverse_jacobian, n, element)
                for v in axes(states, 1)
                    states[v, q] += weight * u[v, i, j, k, element]
                end
            end
            u_q = SVector{Trixi.nvariables(equations)}(view(states, :, q))
            values[:, q] = jq * source.source_terms(u_q, x, t, equations)
        end
        for n in Base.OneTo(num_nodes)
            i, j, k = Tuple(CartesianIndices((num_nodes_1d(dg), num_nodes_1d(dg),
                                              num_nodes_1d(dg)))[n])
            scale = inverse_jacobian_at(inverse_jacobian, n, element)
            for v in axes(values, 1)
                value = zero(eltype(values))
                for q in Base.OneTo(num_points)
                    value += projection[n, q] * values[v, q]
                end
                du[v, i, j, k, element] += scale * value
            end
        end
    end
    return nothing
end

@inline num_nodes_1d(dg::DGSEM) = length(dg.basis.nodes)
@inline inverse_jacobian_at(inverse_jacobian::AbstractVector, n, element) = inverse_jacobian[element]
@inline function inverse_jacobian_at(inverse_jacobian::AbstractArray{<:Any, 4}, n, element)
    return view(inverse_jacobian, :, :, :, element)[n]
end
