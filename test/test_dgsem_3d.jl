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
