# Divergence diagnostics on the hexahedral DGSEM meshes, computed from the nodal
# derivative matrix and the metric terms of each element.

const CurvedDGSEMMesh3D = Union{Trixi.P4estMesh{3}, Trixi.StructuredMesh{3},
                                Trixi.T8codeMesh{3}}

# TreeMesh stores the one-dimensional scaling 2 / h per element
@inline node_jacobian(inverse_jacobian::AbstractVector, i, j, k, element) = inv(inverse_jacobian[element])^3
@inline function node_jacobian(inverse_jacobian::AbstractArray{<:Any, 4}, i, j, k,
                               element)
    return inv(inverse_jacobian[i, j, k, element])
end

function nodal_divergence!(divergence, u, indices, element, mesh::Trixi.TreeMesh{3},
                           dg::DGSEM, cache)
    (; derivative_matrix) = dg.basis
    inverse_jacobian = cache.elements.inverse_jacobian[element]
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        value = zero(eltype(divergence))
        for m in eachnode(dg)
            value += derivative_matrix[i, m] * u[indices[1], m, j, k, element] +
                     derivative_matrix[j, m] * u[indices[2], i, m, k, element] +
                     derivative_matrix[k, m] * u[indices[3], i, j, m, element]
        end
        divergence[i, j, k] = inverse_jacobian * value
    end
    return divergence
end

function nodal_divergence!(divergence, u, indices, element, mesh::CurvedDGSEMMesh3D,
                           dg::DGSEM, cache)
    (; derivative_matrix) = dg.basis
    (; contravariant_vectors, inverse_jacobian) = cache.elements
    num_nodes = Trixi.nnodes(dg)
    contravariant_flux = zeros(eltype(divergence), 3, num_nodes, num_nodes, num_nodes)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        field = SVector(u[indices[1], i, j, k, element], u[indices[2], i, j, k, element],
                        u[indices[3], i, j, k, element])
        for d in 1:3
            Ja = Trixi.get_contravariant_vector(d, contravariant_vectors, i, j, k,
                                                element)
            contravariant_flux[d, i, j, k] = dot(Ja, field)
        end
    end
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        value = zero(eltype(divergence))
        for m in eachnode(dg)
            value += derivative_matrix[i, m] * contravariant_flux[1, m, j, k] +
                     derivative_matrix[j, m] * contravariant_flux[2, i, m, k] +
                     derivative_matrix[k, m] * contravariant_flux[3, i, j, m]
        end
        divergence[i, j, k] = inverse_jacobian[i, j, k, element] * value
    end
    return divergence
end

function divergence_norms(u, indices, mesh::Trixi.AbstractMesh{3}, dg::DGSEM, cache)
    (; weights) = dg.basis
    (; inverse_jacobian) = cache.elements
    num_nodes = Trixi.nnodes(dg)
    divergence = zeros(eltype(u), num_nodes, num_nodes, num_nodes)
    l2norm_squared = zero(eltype(u))
    linfnorm = zero(eltype(u))
    for element in Trixi.eachelement(dg, cache)
        nodal_divergence!(divergence, u, indices, element, mesh, dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            volume = weights[i] * weights[j] * weights[k] *
                     node_jacobian(inverse_jacobian, i, j, k, element)
            l2norm_squared += volume * divergence[i, j, k]^2
            linfnorm = max(linfnorm, abs(divergence[i, j, k]))
        end
    end
    return sqrt(l2norm_squared), linfnorm
end

function Trixi.analyze(::Val{:l2_dive}, du, u, t, mesh::Trixi.AbstractMesh{3},
                       equations::MaxwellEquations3D, dg::DGSEM, cache)
    return divergence_norms(u, electric_field_indices, mesh, dg, cache)[1]
end

function Trixi.analyze(::Val{:linf_dive}, du, u, t, mesh::Trixi.AbstractMesh{3},
                       equations::MaxwellEquations3D, dg::DGSEM, cache)
    return divergence_norms(u, electric_field_indices, mesh, dg, cache)[2]
end

function Trixi.analyze(::Val{:l2_divh}, du, u, t, mesh::Trixi.AbstractMesh{3},
                       equations::MaxwellEquations3D, dg::DGSEM, cache)
    return divergence_norms(u, magnetic_field_indices, mesh, dg, cache)[1]
end

function Trixi.analyze(::Val{:linf_divh}, du, u, t, mesh::Trixi.AbstractMesh{3},
                       equations::MaxwellEquations3D, dg::DGSEM, cache)
    return divergence_norms(u, magnetic_field_indices, mesh, dg, cache)[2]
end

# The passive components are set per element or driven by sources, not by the
# initial condition, so their error against it carries no information: report zero.
function mask_passive_errors(l2_error, linf_error)
    mask = SVector(ntuple(i -> i <= 6 ? one(eltype(l2_error)) : zero(eltype(l2_error)),
                          Val(length(l2_error))))
    return l2_error .* mask, linf_error .* mask
end

function Trixi.calc_error_norms(func, u, t, analyzer, mesh::Trixi.TreeMesh{3},
                                equations::MaxwellEquations3D, initial_condition,
                                dg::DGSEM, cache, cache_analysis)
    return mask_passive_errors(invoke(Trixi.calc_error_norms,
                                      Tuple{Any, Any, Any, Any, Trixi.TreeMesh{3}, Any,
                                            Any, DGSEM, Any, Any},
                                      func, u, t, analyzer, mesh, equations,
                                      initial_condition, dg, cache, cache_analysis)...)
end

function Trixi.calc_error_norms(func, u, t, analyzer, mesh::CurvedDGSEMMesh3D,
                                equations::MaxwellEquations3D, initial_condition,
                                dg::DGSEM, cache, cache_analysis)
    return mask_passive_errors(invoke(Trixi.calc_error_norms,
                                      Tuple{Any, Any, Any, Any, CurvedDGSEMMesh3D, Any,
                                            Any, DGSEM, Any, Any},
                                      func, u, t, analyzer, mesh, equations,
                                      initial_condition, dg, cache, cache_analysis)...)
end
