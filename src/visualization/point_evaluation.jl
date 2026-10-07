"""
    PointEvaluator(points, semi)

Interpolation of a solution to `points`, on the straight-sided or curved
tetrahedra of a `DGMultiMesh` or on the hexahedra of any three-dimensional
`DGSEM` mesh. Each
point is located in its element once; `evaluator(u_ode, semi)` then returns the
state at every point.
"""
struct PointEvaluator{RealT <: Real}
    elements::Vector{Int}
    interpolation::Matrix{RealT}
end

function PointEvaluator(points, semi)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    return PointEvaluator(points, mesh, dg, cache)
end

function PointEvaluator(points, mesh::DGMultiMesh, dg::DGMulti, cache)
    rd = dg.basis
    md = mesh.md
    RealT = eltype(md.xyz[1])

    reference_vertices = ([-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1])
    to_vertices = StartUpDG.vandermonde(rd.element_type, rd.N, reference_vertices...) /
                  rd.VDM

    elements = Int[]
    interpolation = zeros(RealT, length(points), size(rd.VDM, 1))
    for (p, point) in enumerate(points)
        element, reference_coordinates = locate_point(SVector{3, RealT}(point), mesh,
                                                      rd, to_vertices)
        push!(elements, element)
        interpolation[p, :] = StartUpDG.vandermonde(rd.element_type, rd.N,
                                                    map(c -> [c],
                                                        reference_coordinates)...) /
                              rd.VDM
    end
    return PointEvaluator(elements, interpolation)
end

# Barycentric coordinates of `point` with respect to the vertices of `element`.
function vertex_barycentric(point, md, to_vertices, element)
    coordinates = ntuple(d -> to_vertices * view(md.xyz[d], :, element), 3)
    vertex(v) = SVector(coordinates[1][v], coordinates[2][v], coordinates[3][v])
    v1 = vertex(1)
    edges = hcat(vertex(2) - v1, vertex(3) - v1, vertex(4) - v1)
    return edges \ (point - v1)
end

function locate_point(point, mesh::DGMultiMesh, rd, to_vertices; tolerance = 1.0e-10)
    md = mesh.md
    for element in Base.OneTo(md.num_elements)
        lambda = vertex_barycentric(point, md, to_vertices, element)
        if all(lambda .>= -tolerance) && sum(lambda) <= 1 + tolerance
            return element, 2 * lambda .- 1
        end
    end
    throw(ArgumentError("point $point lies outside the mesh"))
end

# Curved tetrahedra: candidates from the vertex map, then Newton iteration on the
# polynomial element mapping.
function locate_point(point, mesh::CurvedDGMultiMesh, rd, to_vertices;
                      tolerance = 1.0e-10, margin = 0.25)
    md = mesh.md
    for element in Base.OneTo(md.num_elements)
        lambda = vertex_barycentric(point, md, to_vertices, element)
        (all(lambda .>= -margin) && sum(lambda) <= 1 + margin) || continue
        coordinates = hcat(view(md.xyz[1], :, element), view(md.xyz[2], :, element),
                           view(md.xyz[3], :, element))
        rst = 2 * lambda .- 1
        converged = false
        for _ in 1:20
            reference = map(c -> [c], Tuple(rst))
            basis = StartUpDG.vandermonde(rd.element_type, rd.N, reference...) / rd.VDM
            derivatives = map(V -> V / rd.VDM,
                              StartUpDG.grad_vandermonde(rd.element_type, rd.N,
                                                         reference...))
            x = SVector{3}(basis * coordinates)
            jacobian = hcat(ntuple(j -> SVector{3}(derivatives[j] * coordinates), 3)...)
            step = jacobian \ (point - x)
            rst += step
            if norm(step) < tolerance
                converged = true
                break
            end
        end
        if converged && all(rst .>= -1 - 1.0e-8) && sum(rst) <= -1 + 1.0e-8
            return element, rst
        end
    end
    throw(ArgumentError("point $point lies outside the mesh"))
end

function (evaluator::PointEvaluator)(u_ode, semi)
    mesh, equations, dg, _ = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)
    return evaluate_points(evaluator, u, mesh, equations, dg)
end

function evaluate_points(evaluator, u, mesh::DGMultiMesh, equations, dg)
    return map(eachindex(evaluator.elements)) do p
        element = evaluator.elements[p]
        sum(evaluator.interpolation[p, i] * u[i, element] for i in axes(u, 1))
    end
end

# DGSEM meshes: invert the polynomial element mapping by Newton iteration, then
# interpolate with the tensor product Lagrange basis.
function PointEvaluator(points, mesh::Trixi.AbstractMesh{3}, dg::DGSEM, cache)
    (; node_coordinates) = cache.elements
    nodes = dg.basis.nodes
    RealT = eltype(node_coordinates)
    num_nodes = length(nodes)
    weights = Trixi.barycentric_weights(nodes)
    derivative = Trixi.polynomial_derivative_matrix(nodes)

    elements = Int[]
    interpolation = zeros(RealT, length(points), num_nodes^3)
    for (p, point) in enumerate(points)
        element, xi = locate_point(SVector{3, RealT}(point), node_coordinates, nodes,
                                   weights, derivative)
        push!(elements, element)
        basis = ntuple(d -> Trixi.lagrange_interpolating_polynomials(xi[d], nodes,
                                                                     weights), 3)
        n = 0
        for k in Base.OneTo(num_nodes), j in Base.OneTo(num_nodes),
            i in Base.OneTo(num_nodes)

            n += 1
            interpolation[p, n] = basis[1][i] * basis[2][j] * basis[3][k]
        end
    end
    return PointEvaluator(elements, interpolation)
end

function locate_point(point, node_coordinates, nodes, weights, derivative;
                      tolerance = 1.0e-10)
    num_nodes = length(nodes)
    for element in axes(node_coordinates, 5)
        coordinates = view(node_coordinates, :, :, :, :, element)
        inside_box = all(d -> minimum(view(coordinates, d, :, :, :)) - tolerance <=
                              point[d] <=
                              maximum(view(coordinates, d, :, :, :)) + tolerance, 1:3)
        inside_box || continue

        xi = zero(SVector{3, eltype(point)})
        converged = false
        for _ in 1:20
            basis = ntuple(d -> Trixi.lagrange_interpolating_polynomials(xi[d], nodes,
                                                                         weights), 3)
            dbasis = ntuple(d -> derivative' * basis[d], 3)
            x = zero(point)
            jacobian = zero(SMatrix{3, 3, eltype(point)})
            for k in Base.OneTo(num_nodes), j in Base.OneTo(num_nodes),
                i in Base.OneTo(num_nodes)

                node = SVector{3}(view(coordinates, :, i, j, k))
                x += basis[1][i] * basis[2][j] * basis[3][k] * node
                jacobian += node *
                            SVector(dbasis[1][i] * basis[2][j] * basis[3][k],
                                    basis[1][i] * dbasis[2][j] * basis[3][k],
                                    basis[1][i] * basis[2][j] * dbasis[3][k])'
            end
            step = jacobian \ (point - x)
            xi += step
            if norm(step) < tolerance
                converged = true
                break
            end
        end
        if converged && all(abs.(xi) .<= 1 + 1.0e-8)
            return element, xi
        end
    end
    throw(ArgumentError("point $point lies outside the mesh"))
end

function evaluate_points(evaluator, u, mesh::Trixi.AbstractMesh{3}, equations,
                         dg::DGSEM)
    return map(eachindex(evaluator.elements)) do p
        element = evaluator.elements[p]
        value = zero(Trixi.get_node_vars(u, equations, dg, 1, 1, 1, element))
        n = 0
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            n += 1
            value += evaluator.interpolation[p, n] *
                     Trixi.get_node_vars(u, equations, dg, i, j, k, element)
        end
        value
    end
end
