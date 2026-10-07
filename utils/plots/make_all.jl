# Regenerates every figure in docs/figures.
for script in ("plot_cavity.jl", "plot_fresnel.jl", "plot_tfsf.jl", "plot_dipole.jl",
               "plot_dipole_p4est.jl",
               "plot_pml.jl", "plot_dielectric_sphere.jl", "plot_pec_sphere.jl")
    include(joinpath(@__DIR__, script))
end
