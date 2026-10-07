# Shared setup for the figure scripts. Run from the package root with
#   julia --project=utils/plots utils/plots/<script>.jl
# after `Pkg.develop(path = ".")` in that environment.

using Trixi
using TrixiMaxwell
using CairoMakie
using StaticArrays: SVector
using LinearAlgebra: norm

const EXAMPLES_DIR = pkgdir(TrixiMaxwell, "examples", "dgmulti_3d")
const FIGURES_DIR = pkgdir(TrixiMaxwell, "docs", "figures")

elixir(name) = joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_$(name).jl")

# Runs an elixir in `Main` and returns its solution and semidiscretization;
# other elixir variables are read with `elixir_var(:name)`. The lookup goes
# through `invokelatest` because the bindings are created after this code was
# compiled.
function run_elixir(name; kwargs...)
    trixi_include(elixir(name); kwargs...)
    return (; sol = elixir_var(:sol), semi = elixir_var(:semi))
end

elixir_var(name) = Base.invokelatest(getglobal, Main, name)

# Heatmap of one variable on a plane through a 3D DGMulti solution. Uses the
# triangulation of Trixi's Makie extension directly so that the color range is
# shared between panels.
function slice_heatmap!(ax, u, semi, variable; slice = :xz, point = (0.0, 0.0, 0.0),
                        colorrange = nothing, colormap = Reverse(:RdBu),
                        plot_mesh = false)
    ext = Base.get_extension(Trixi, :TrixiMakieExt)
    pds = PlotData2D(u, semi; slice, point)[variable]
    values = getindex.(ext.global_plotting_triangulation_makie(pds).position, 3)
    triangulation = ext.global_plotting_triangulation_makie(pds;
                                                            set_z_coordinate_zero = true)
    if colorrange === nothing
        limit = maximum(abs, values)
        colorrange = (-limit, limit)
    end
    plt = mesh!(ax, triangulation; color = values, colormap, colorrange,
                shading = NoShading)
    if plot_mesh
        lines!(ax, ext.convert_PlotData2D_to_mesh_Points(pds; set_z_coordinate_zero = true);
               color = :lightgrey)
    end
    return plt
end

function save_figure(name, fig)
    path = joinpath(FIGURES_DIR, name)
    save(path, fig; px_per_unit = 2)
    println("saved ", path)
    return path
end

# Heatmap of one variable on an axis-aligned plane through a DGSEM solution on
# a Cartesian hexahedral mesh (TreeMesh, StructuredMesh or P4estMesh without a
# mapping). Each element cut by the plane is sampled on a plotting lattice.
function hex_slice_heatmap!(ax, u_ode, semi, variable; slice = :xz,
                            point = (0.0, 0.0, 0.0), colorrange = nothing,
                            colormap = Reverse(:RdBu), nvisnodes = nothing,
                            plot_mesh = false)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)
    index = findfirst(==(variable), Trixi.varnames(cons2cons, equations))
    orientation_x, orientation_y, slice_dimension = slice == :xz ? (1, 3, 2) :
                                                    slice == :xy ? (1, 2, 3) : (2, 3, 1)
    plane = point[slice_dimension]
    nodes = dg.basis.nodes
    num_nodes = length(nodes)
    nvis = nvisnodes === nothing ? 2 * num_nodes : nvisnodes
    weights = Trixi.barycentric_weights(nodes)
    plotting = range(-1.0, 1.0, length = nvis)
    # built point by point: the lattice contains the element end points, which the
    # matrix routine cannot evaluate at
    to_plotting = reduce(vcat,
                         (Trixi.lagrange_interpolating_polynomials(x, nodes, weights)'
                          for x in plotting))
    node_coordinates = cache.elements.node_coordinates
    upper = maximum(view(node_coordinates, slice_dimension, :, :, :, :))

    vertices = zeros(0, 2)
    faces = zeros(Int, 0, 3)
    values = Float64[]
    outlines = Point2f[]
    for element in axes(node_coordinates, 5)
        lower = minimum(view(node_coordinates, slice_dimension, :, :, :, element))
        top = maximum(view(node_coordinates, slice_dimension, :, :, :, element))
        inside = lower <= plane < top || (plane == top && top == upper)
        inside || continue
        eta = 2 * (plane - lower) / (top - lower) - 1
        basis = Trixi.lagrange_interpolating_polynomials(eta, nodes, weights)
        data = Array(u[index, :, :, :, element])
        shape = ntuple(d -> d == slice_dimension ? num_nodes : 1, 3)
        sliced = dropdims(sum(reshape(basis, shape) .* data, dims = slice_dimension),
                          dims = slice_dimension)
        function axis(orientation)
            [node_coordinates[orientation,
                              ntuple(d -> d == orientation ? i : 1, 3)...,
                              element] for i in 1:num_nodes]
        end
        xs = to_plotting * axis(orientation_x)
        ys = to_plotting * axis(orientation_y)
        plotted = to_plotting * sliced * to_plotting'
        append!(outlines,
                Point2f.([(xs[1], ys[1]), (xs[end], ys[1]), (xs[end], ys[end]),
                             (xs[1], ys[end]), (xs[1], ys[1]), (NaN, NaN)]))
        offset = size(vertices, 1)
        vertices = vcat(vertices, [repeat(xs, nvis) repeat(ys, inner = nvis)])
        append!(values, vec(plotted))
        for j in 1:(nvis - 1), i in 1:(nvis - 1)
            a = offset + i + (j - 1) * nvis
            faces = vcat(faces, [a a+1 a+nvis; a+1 a+nvis+1 a+nvis])
        end
    end
    if colorrange === nothing
        limit = maximum(abs, values)
        colorrange = (-limit, limit)
    end
    plt = mesh!(ax, vertices, faces; color = values, colormap, colorrange,
                shading = NoShading)
    plot_mesh && lines!(ax, outlines; color = :lightgrey, linewidth = 0.5)
    return plt
end
