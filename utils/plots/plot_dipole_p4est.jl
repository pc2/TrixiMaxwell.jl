# Hertzian dipole on P4estMesh with adaptive refinement: the same figure as for tetrahedra.
include("common.jl")

trixi_include(pkgdir(TrixiMaxwell, "examples", "p4est_3d_dgsem",
                     "elixir_maxwell_3d_dipole.jl"))
sol, semi = elixir_var(:sol), elixir_var(:semi)
dipole_field = elixir_var(:dipole_field)
equations = elixir_var(:equations)

fig = Figure(size = (1000, 420))
ax = Axis(fig[1, 1], aspect = DataAspect(), title = "Ez at t = $(sol.t[end])",
          xlabel = "x", ylabel = "z")
plt = hex_slice_heatmap!(ax, sol.u[end], semi, "Ez"; slice = :xz,
                         point = (0.0, 0.0, 0.0), colorrange = (-1.5, 1.5),
                         plot_mesh = true)
Colorbar(fig[1, 2], plt)

xs = range(0.3, 0.95, length = 60)
points = [SVector(x, 0.0, 0.0) for x in xs]
numerical = PointEvaluator(points, semi)(sol.u[end], semi)
exact = [dipole_field(x, sol.t[end], equations) for x in points]
ax2 = Axis(fig[1, 3], title = "Ez along the x axis", xlabel = "x", ylabel = "Ez")
lines!(ax2, xs, [u[3] for u in exact]; color = :black, label = "exact")
scatter!(ax2, xs, [u[3] for u in numerical]; color = :red, markersize = 6, label = "DG")
axislegend(ax2)
save_figure("dipole_p4est.png", fig)
