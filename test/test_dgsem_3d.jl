module TestExamplesDGSEM3D

using Test
using Trixi
using TrixiMaxwell

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(TrixiMaxwell, "examples")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGSEM 3D" begin
#! format: noindent

@trixi_testset "tree_3d_dgsem/elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "tree_3d_dgsem",
                                 "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            2.802969543936305e-17,
                            2.7753392139966613e-17,
                            0.00014027374758763525,
                            9.21647934455528e-5,
                            9.216479344555087e-5,
                            1.5827546384527813e-17
                        ],
                        linf=[
                            4.174319517227513e-16,
                            4.2629657427462536e-16,
                            0.0011771469476341556,
                            0.0005978533018992649,
                            0.0005978533018992094,
                            2.3120348744330817e-16
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "structured_3d_dgsem/elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "structured_3d_dgsem",
                                 "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            2.282045947896377e-16,
                            2.2796942639854023e-16,
                            0.00014027374758770003,
                            9.216479344557848e-5,
                            9.216479344557554e-5,
                            8.798072749662231e-16
                        ],
                        linf=[
                            1.0357340681663061e-14,
                            1.0174910257163313e-14,
                            0.001177146947636376,
                            0.000597853301942175,
                            0.0005978533019426191,
                            3.6543742410353625e-14
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "structured_3d_dgsem/elixir_maxwell_3d_cavity_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "structured_3d_dgsem",
                                 "elixir_maxwell_3d_cavity_curved.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            8.225401493908235e-5,
                            8.225401493908227e-5,
                            0.0002542020296862264,
                            0.00024081644139219266,
                            0.00024081644139219247,
                            5.969201781517723e-5
                        ],
                        linf=[
                            0.0007563723826874326,
                            0.0007563723826871896,
                            0.002249697768541614,
                            0.002012000472957798,
                            0.0020120004729574648,
                            0.0005969402650691219
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            3.257656230930607e-16,
                            3.2609130750042396e-16,
                            0.00014027374758763175,
                            9.21647934455535e-5,
                            9.216479344554801e-5,
                            7.531893065353821e-16
                        ],
                        linf=[
                            2.13833552899782e-14,
                            3.081738143218611e-14,
                            0.0011771469476381524,
                            0.0005978533019454502,
                            0.000597853301934903,
                            4.5873492015260104e-14
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_cavity_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_cavity_nonconforming.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            0.0002464737524758418,
                            0.0002464737524758429,
                            0.0016237005524070275,
                            0.001577647967180522,
                            0.0015776479671805162,
                            8.64630250612e-5
                        ],
                        linf=[
                            0.0052781246886508424,
                            0.0052781246886540656,
                            0.01399807954346255,
                            0.006078255766559067,
                            0.00607825576655798,
                            0.00191660812216715
                        ])
    @test Trixi.nelements(solver, semi.cache) == 120
    @test Trixi.nmortars(semi.cache.mortars) == 36
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "t8code_3d_dgsem/elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "t8code_3d_dgsem",
                                 "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            3.644167304557558e-16,
                            3.676747880509144e-16,
                            0.00014027374758764747,
                            9.216479344555875e-5,
                            9.216479344555533e-5,
                            7.56834047750736e-16
                        ],
                        linf=[
                            2.312014365701868e-14,
                            3.069350913111878e-14,
                            0.0011771469476387075,
                            0.000597853301934681,
                            0.0005978533019442844,
                            4.8938331221371796e-14
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_dipole.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_dipole.jl"))
    using TrixiMaxwell: electric_field, magnetic_field
    using LinearAlgebra: norm
    probes = [SVector(0.5, 0.0, 0.0), SVector(0.0, 0.5, 0.0), SVector(0.0, 0.0, 0.5),
        SVector(0.35, 0.35, 0.0), SVector(0.3, 0.2, 0.4), SVector(-0.4, 0.1, -0.5)]
    evaluator = PointEvaluator(probes, semi)
    numerical = evaluator(sol.u[end], semi)
    exact = [dipole_field(x, sol.t[end], equations) for x in probes]
    @test norm(reduce(vcat, numerical) - reduce(vcat, exact)) <
          0.01 * norm(reduce(vcat, exact))
    # on the dipole axis the far field vanishes
    @test norm(magnetic_field(numerical[3])) < 0.01 * norm(magnetic_field(numerical[1]))
    @test norm(electric_field(numerical[3])) < 0.5 * norm(electric_field(numerical[1]))
    # refined once where the pulse is and twice around the source
    @test 512 < Trixi.nelements(solver, semi.cache) < 4096 + 512
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_pml.jl" begin
    using Trixi, TrixiMaxwell
    # coarser mesh with a two-cell layer and a source the hexahedra resolve;
    # after the pulse has passed, the energy left in the box measures the
    # reflections from the layer
    remaining_energy(sol, semi) = Trixi.integrate(energy_total, sol.u[end], semi,
                                                  normalize = false)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_pml.jl"),
                        trees_per_dimension=(4, 4, 4), pml_thickness=0.75,
                        dipole=HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2,
                                              GaussianPulse(0.4; delay = 1.4)))
    @test Trixi.nvariables(equations) == 12
    energy_pml = remaining_energy(sol, semi)
    u = Trixi.wrap_array(sol.u[end], semi)
    node_coordinates = semi.cache.elements.node_coordinates
    # auxiliary fields vanish in elements that lie entirely in the physical region
    inside(element) = all(maximum(abs, view(node_coordinates, d, :, :, :, element)) <
                          0.75
                          for d in 1:3)
    @test count(element -> inside(element) &&
                    any(!iszero, view(u, 7:12, :, :, :, element)),
                axes(u, 5)) == 0
    @test any(!iszero, view(u, 7:12, :, :, :, :))
    @test maximum(abs, view(u, 7:12, :, :, :, :)) < 10
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # Silver-Mueller alone on the same domain reflects far more
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_pml.jl"),
                        trees_per_dimension=(4, 4, 4),
                        dipole=HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.2,
                                              GaussianPulse(0.4; delay = 1.4)),
                        equations=MaxwellEquations3D(),
                        source_terms=ProjectedSourceTerms(dipole, MaxwellEquations3D(),
                                                          DGSEM(polydeg = 2,
                                                                surface_flux = flux_upwind)))
    energy_silver_mueller = remaining_energy(sol, semi)
    @test energy_pml < 0.25 * energy_silver_mueller
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_fresnel.jl"))
    errors = analysis_callback(sol)
    # half the degrees of freedom of the tetrahedral mesh at the same cell size
    @test maximum(errors.l2[1:6]) < 5e-2
    @test all(iszero, errors.l2[7:9]) && all(iszero, errors.linf[7:9])
    # passive material components stay fixed
    u = Trixi.wrap_array(sol.u[end], semi)
    @test all(x -> x in (1.0, 4.0), view(u, 7, :, :, :, :))
    @test all(==(1.0), view(u, 8, :, :, :, :)) && all(iszero, view(u, 9, :, :, :, :))
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel.jl (interface energy conservation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_fresnel.jl"),
                        surface_flux=FluxUpwindPenalty(0.0),
                        boundary_conditions=(;
                                             x_neg = boundary_condition_perfect_electric_conductor,
                                             x_pos = boundary_condition_perfect_electric_conductor),
                        tspan=(0.0, 0.6))
    u = sol.u[end]
    du = similar(u)
    Trixi.rhs_hyperbolic!(du, u, semi, sol.t[end])
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    energy_rate = Trixi.analyze(Trixi.entropy_timederivative,
                                Trixi.wrap_array(du, semi), Trixi.wrap_array(u, semi),
                                sol.t[end], mesh, equations, solver, cache)
    energy = Trixi.integrate(energy_total, u, semi, normalize = false)
    @test abs(energy_rate) < 1e-12 * energy
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel.jl (mortars across the jump)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_fresnel.jl"),
                        refine_interface=true,
                        surface_flux=FluxUpwindPenalty(0.0),
                        boundary_conditions=(;
                                             x_neg = boundary_condition_perfect_electric_conductor,
                                             x_pos = boundary_condition_perfect_electric_conductor),
                        tspan=(0.0, 0.6))
    @test Trixi.nmortars(semi.cache.mortars) > 0
    u = sol.u[end]
    du = similar(u)
    Trixi.rhs_hyperbolic!(du, u, semi, sol.t[end])
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    energy_rate = Trixi.analyze(Trixi.entropy_timederivative,
                                Trixi.wrap_array(du, semi), Trixi.wrap_array(u, semi),
                                sol.t[end], mesh, equations, solver, cache)
    energy = Trixi.integrate(energy_total, u, semi, normalize = false)
    @test abs(energy_rate) < 1e-12 * energy
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel.jl (mortars, upwind)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_fresnel.jl"),
                        refine_interface=true)
    @test Trixi.nmortars(semi.cache.mortars) > 0
    errors = analysis_callback(sol)
    @test maximum(errors.l2[1:6]) < 5e-2
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel.jl (convergence)" begin
    using Trixi, TrixiMaxwell
    eocs, _ = Trixi.convergence_test(@__MODULE__,
                                     joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                              "elixir_maxwell_3d_fresnel.jl"),
                                     3; polydeg = 2, cfl = 0.3,
                                     initial_refinement_level = 1)
    # Ey and Hz carry the pulse, the other components vanish exactly on hexahedra
    @test all(eocs[:l2][end, [2, 6]] .> 2.75)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_tfsf.jl" begin
    using Trixi, TrixiMaxwell
    # the pulse centre is inside the box at t = 1.5
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_tfsf.jl"),
                        tspan=(0.0, 1.5))
    mesh_, equations_, solver_, cache_ = Trixi.mesh_equations_solver_cache(semi)
    @test count(!iszero, surface_flux.signs) == 6 * 16
    @test occursin("96 interfaces", repr(surface_flux))
    u = Trixi.wrap_array(sol.u[end], semi)
    (; weights) = solver_.basis
    (; inverse_jacobian, node_coordinates) = cache_.elements
    energy_in = 0.0
    energy_out = 0.0
    for element in Trixi.eachelement(solver_, cache_)
        energy = 0.0
        for k in eachnode(solver_), j in eachnode(solver_), i in eachnode(solver_)
            u_node = Trixi.get_node_vars(u, equations_, solver_, i, j, k, element)
            energy += weights[i] * weights[j] * weights[k] *
                      energy_total(u_node, equations_) /
                      inverse_jacobian[i, j, k, element]
        end
        if is_total_field(TrixiMaxwell.element_centroid(node_coordinates, element))
            energy_in += energy
        else
            energy_out += energy
        end
    end
    # exact energy of the pulse: cross-section 1 times the integral of exp(-2 (x / w)^2)
    # hexahedra carry half the degrees of freedom of the tetrahedra at this
    # cell size, so the leakage is larger than on tets
    @test energy_in≈0.25 * sqrt(pi / 2) rtol=2e-2
    @test energy_out < 1e-4 * energy_in
    peak_energy = Trixi.integrate(energy_total, sol.u[end], semi)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # the pulse has left the box through its far face and nothing remains
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_tfsf.jl"))
    @test Trixi.integrate(energy_total, sol.u[end], semi) < 1e-7 * peak_energy
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_pml_amr.jl" begin
    using Trixi, TrixiMaxwell
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_pml_amr.jl"),
                        trees_per_dimension=(4, 4, 4), pml_thickness=0.75,
                        tspan=(0.0, 2.5))
    @test Trixi.nmortars(semi.cache.mortars) > 0
    u = Trixi.wrap_array(sol.u[end], semi)
    @test any(!iszero, view(u, 7:12, :, :, :, :))
    @test maximum(abs, view(u, 7:12, :, :, :, :)) < 10
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_fresnel_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_fresnel_amr.jl"))
    @test Trixi.nmortars(semi.cache.mortars) > 0
    errors = analysis_callback(sol)
    @test maximum(errors.l2[1:6]) < 5e-2
    # refinement and coarsening project the material components to rounding
    u = Trixi.wrap_array(sol.u[end], semi)
    @test all(x -> isapprox(x, 1.0; atol = 1e-10) || isapprox(x, 4.0; atol = 1e-10),
              view(u, 7, :, :, :, :))
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "p4est_3d_dgsem/elixir_maxwell_3d_tfsf_amr.jl" begin
    using Trixi, TrixiMaxwell
    @test_trixi_include(joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                                 "elixir_maxwell_3d_tfsf_amr.jl"),
                        tspan=(0.0, 1.5))
    @test Trixi.nmortars(semi.cache.mortars) > 0
    # the box faces are refined once on both sides: 6 faces of 4 x 4 elements
    @test count(!iszero, surface_flux.signs) == 6 * 16 * 4
    mesh_, equations_, solver_, cache_ = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(sol.u[end], semi)
    (; weights) = solver_.basis
    (; inverse_jacobian, node_coordinates) = cache_.elements
    energy_in = 0.0
    energy_out = 0.0
    for element in Trixi.eachelement(solver_, cache_)
        energy = 0.0
        for k in eachnode(solver_), j in eachnode(solver_), i in eachnode(solver_)
            u_node = Trixi.get_node_vars(u, equations_, solver_, i, j, k, element)
            energy += weights[i] * weights[j] * weights[k] *
                      energy_total(u_node, equations_) /
                      inverse_jacobian[i, j, k, element]
        end
        if is_total_field(TrixiMaxwell.element_centroid(node_coordinates, element))
            energy_in += energy
        else
            energy_out += energy
        end
    end
    @test energy_in≈0.25 * sqrt(pi / 2) rtol=2e-2
    @test energy_out < 1e-4 * energy_in
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "cavity convergence on all mesh types" begin
    using Trixi, TrixiMaxwell
    # Ez, Hx, Hy carry the mode; Ex, Ey, Hz are zero in the exact solution
    mode_components = (3, 4, 5)
    polydeg = 2
    common = (; polydeg, cfl = 0.25, tspan = (0.0, 0.5))
    cases = (("tree_3d_dgsem", "elixir_maxwell_3d_cavity.jl",
              (; initial_refinement_level = 2), polydeg + 0.75),
             ("structured_3d_dgsem", "elixir_maxwell_3d_cavity.jl",
              (; cells_per_dimension = (4, 4, 4)), polydeg + 0.75),
             # curved elements resolve the warp only gradually at these cell counts
             ("structured_3d_dgsem", "elixir_maxwell_3d_cavity_curved.jl",
              (; cells_per_dimension = (4, 4, 4)), polydeg + 0.5),
             ("p4est_3d_dgsem", "elixir_maxwell_3d_cavity.jl",
              (; trees_per_dimension = (1, 1, 1), initial_refinement_level = 2),
              polydeg + 0.75),
             ("p4est_3d_dgsem", "elixir_maxwell_3d_cavity_nonconforming.jl",
              (; trees_per_dimension = (1, 1, 1), initial_refinement_level = 2),
              polydeg + 0.75),
             ("t8code_3d_dgsem", "elixir_maxwell_3d_cavity.jl",
              (; trees_per_dimension = (1, 1, 1), initial_refinement_level = 2),
              polydeg + 0.75))
    for (directory, elixir, resolution, expected_order) in cases
        eocs, _ = Trixi.convergence_test(@__MODULE__,
                                         joinpath(EXAMPLES_DIR, directory, elixir), 3;
                                         common..., resolution...)
        # the coarsest pair is pre-asymptotic, judge the finest refinement step
        eoc_finest = eocs[:l2][end, :]
        @test all(eoc_finest[c] > expected_order for c in mode_components)
    end
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
