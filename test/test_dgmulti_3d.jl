module TestExamplesDGMulti3D

using Test
using Trixi
using TrixiMaxwell

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(TrixiMaxwell, "examples", "dgmulti_3d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGMulti 3D" begin
#! format: noindent

@trixi_testset "elixir_maxwell_3d_cavity_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity_curved.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            0.0010091560891843157,
                            0.0010036108487806695,
                            0.0025058614712118767,
                            0.002031940528267993,
                            0.0020835764266782597,
                            0.000589928470461924
                        ],
                        linf=[
                            0.011110261939409474,
                            0.015067271450955258,
                            0.04272391568467007,
                            0.016574022972575944,
                            0.023595499357964768,
                            0.0055817353782471645
                        ])
    @test mesh isa DGMultiMesh{3, Trixi.NonAffine}
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_periodic.jl"),
                        tspan=(0.0, 0.1),
                        l2=[
                            0.004026282519168456,
                            0.013830965800949144,
                            0.0013561586607982563,
                            0.004587291244711452,
                            0.0013306309232592418,
                            0.013525193474118773
                        ],
                        linf=[
                            0.018962436875544983,
                            0.09160119756625695,
                            0.012413018884185844,
                            0.026963233136926276,
                            0.008512663708931854,
                            0.08438493308564704
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            0.0007793129472875638,
                            0.0007826285312641698,
                            0.0020419344903954604,
                            0.0015324399388593466,
                            0.001560580391670923,
                            0.0004480688735142423
                        ],
                        linf=[
                            0.007836853414515583,
                            0.011000334046308386,
                            0.02857835700820057,
                            0.007525831485687683,
                            0.011446976834034457,
                            0.00534208660038957
                        ])
    mktempdir() do dir
        files = write_solution_vtk(sol.u[end], semi, joinpath(dir, "cavity"))
        @test basename.(files) == ["cavity.vtu"]
        files = write_solution_vtk(sol.u[end], semi, joinpath(dir, "energy");
                                   solution_variables = cons2prim)
        @test isfile(first(files))
        collection = write_solution_vtk(sol.u, sol.t, semi, joinpath(dir, "series"))
        @test isfile(collection)
        @test count(==('<'), read(collection, String)) >= 4 + length(sol.t)
        @test_throws ArgumentError write_solution_vtk(sol.u, sol.t[1:1], semi,
                                                      joinpath(dir, "bad"))
    end

    # VTK output during the simulation, by time and by step count
    mktempdir() do dir
        callback_dt = SaveVtkCallback(dt = 0.05, output_directory = dir,
                                      filename = "by_time")
        solve(ode, CarpenterKennedy2N54(williamson_condition = false);
              dt = 1.0, ode_default_options()...,
              callback = CallbackSet(stepsize_callback, callback_dt))
        entries = callback_dt.affect!.affect!.entries
        @test first(entries)[1] == 0.0
        @test last(entries)[1] ≈ 0.2
        @test length(entries) >= 5
        @test all(isfile(joinpath(dir, file)) for (_, file) in entries)
        @test isfile(joinpath(dir, "by_time.pvd"))
        @test occursin("SaveVtkCallback(dt=0.05)", sprint(show, callback_dt))

        callback_interval = SaveVtkCallback(interval = 5, output_directory = dir,
                                            filename = "by_step",
                                            solution_variables = cons2cons)
        solve(ode, CarpenterKennedy2N54(williamson_condition = false);
              dt = 1.0, ode_default_options()...,
              callback = CallbackSet(stepsize_callback, callback_interval))
        @test length(callback_interval.affect!.entries) >= 3
        @test occursin("SaveVtkCallback(interval=5)", sprint(show, callback_interval))
        @test occursin("output directory",
                       sprint(show, MIME"text/plain"(), callback_interval))
        @test_throws ArgumentError SaveVtkCallback(interval = 2, dt = 0.1)
    end
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (central flux, energy conservation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        surface_flux=FluxUpwindPenalty(0.0), tspan=(0.0, 0.5),
                        l2=[
                            0.0031108530820131036,
                            0.003106392979200707,
                            0.003633740045071118,
                            0.003699456386487258,
                            0.00377399785887293,
                            0.003049410562590979
                        ],
                        linf=[
                            0.05717224250663998,
                            0.04801641087014596,
                            0.04885040982352329,
                            0.0371122790946351,
                            0.039151855392283566,
                            0.03687370171164024
                        ])

    # semidiscrete energy derivative vanishes to rounding
    u = sol.u[end]
    du = similar(u)
    Trixi.rhs_hyperbolic!(du, u, semi, sol.t[end])
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    energy_rate = Trixi.analyze(Trixi.entropy_timederivative,
                                Trixi.wrap_array(du, semi), Trixi.wrap_array(u, semi),
                                sol.t[end], mesh, equations, solver, cache)
    energy = Trixi.integrate(energy_total, u, semi, normalize = false)
    @test abs(energy_rate) < 1e-12 * energy

    # fully discrete drift is the RK error and shrinks with the time step
    energy_start = Trixi.integrate(energy_total, sol.u[1], semi, normalize = false)
    drift_coarse = abs(energy - energy_start) / energy_start
    @test drift_coarse < 1e-6

    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                  surface_flux = FluxUpwindPenalty(0.0), tspan = (0.0, 0.5), cfl = 0.25)
    energy_fine = Trixi.integrate(energy_total, sol.u[end], semi, normalize = false)
    drift_fine = abs(energy_fine - energy_start) / energy_start
    @test drift_fine < drift_coarse / 8
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (lossy medium)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                        equations=MaxwellEquations3D(sigma = 0.5),
                        source_terms=source_terms_conductivity, tspan=(0.0, 0.5))
    # same error level as the lossless mode, energy follows the damped exact solution
    @test maximum(analysis_callback(sol).l2) < 3e-3
    energy_end = Trixi.integrate(energy_total, sol.u[end], semi)
    energy_exact = Trixi.integrate(energy_total,
                                   Trixi.compute_coefficients(sol.t[end], semi),
                                   semi)
    @test energy_end < 0.85 * Trixi.integrate(energy_total, sol.u[1], semi)
    @test isapprox(energy_end, energy_exact, rtol = 1e-3)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # central flux: the semidiscrete energy rate is exactly the Ohmic loss
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity.jl"),
                  equations = MaxwellEquations3D(sigma = 0.5),
                  source_terms = source_terms_conductivity,
                  surface_flux = FluxUpwindPenalty(0.0), tspan = (0.0, 0.3))
    u = sol.u[end]
    du = similar(u)
    Trixi.rhs_hyperbolic!(du, u, semi, sol.t[end])
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    energy_rate = Trixi.analyze(Trixi.entropy_timederivative,
                                Trixi.wrap_array(du, semi), Trixi.wrap_array(u, semi),
                                sol.t[end], mesh, equations, solver, cache)
    ohmic_loss = Trixi.integrate(u, semi, normalize = false) do u_node, equations
        E = TrixiMaxwell.electric_field(u_node)
        return conductivity(u_node, equations) * sum(abs2, E)
    end
    @test isapprox(energy_rate, -ohmic_loss, rtol = 1e-12)
end

@trixi_testset "elixir_maxwell_3d_cavity.jl (convergence)" begin
    using Trixi, TrixiMaxwell
    # Ez, Hx, Hy carry the mode; Ex, Ey, Hz are zero in the exact solution
    mode_components = (3, 4, 5)
    for polydeg in (1, 2)
        eocs, _ = Trixi.convergence_test(@__MODULE__,
                                         joinpath(EXAMPLES_DIR,
                                                  "elixir_maxwell_3d_cavity.jl"),
                                         3; polydeg, cells_per_dimension = (2, 2, 2),
                                         cfl = 0.1, tspan = (0.0, 0.5))
        # the coarsest pair is pre-asymptotic, judge the finest refinement step
        eoc_finest = eocs[:l2][end, :]
        @test all(eoc_finest[c] > polydeg + 0.75 for c in mode_components)
    end
end

@trixi_testset "elixir_maxwell_3d_silver_mueller.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_silver_mueller.jl"),
                        l2=[
                            0.00117439020493943,
                            0.001708558869573452,
                            0.0010304124230123169,
                            0.0015951257294803507,
                            0.0014479087538807533,
                            0.002192990311656334
                        ],
                        linf=[
                            0.006127871499208194,
                            0.012858598940748847,
                            0.0067441046663994884,
                            0.010557906689524062,
                            0.008339618562718715,
                            0.018482229839108192
                        ])
    # the pulse has left the box, what remains is projection error, not reflection
    energy_start = Trixi.integrate(energy_total, sol.u[1], semi)
    energy_end = Trixi.integrate(energy_total, sol.u[end], semi)
    @test energy_end < 2e-4 * energy_start
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity_gambit.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity_gambit.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            0.0006415131797157973,
                            0.0005944706167109416,
                            0.001285024946453144,
                            0.0013849052465082,
                            0.0013121243449321467,
                            0.0004951117641383888
                        ],
                        linf=[
                            0.011927041307572706,
                            0.01182077437263471,
                            0.03945812099332727,
                            0.027411664146136775,
                            0.02361227280134434,
                            0.01664116188118837
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_cavity_gmsh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_cavity_gmsh.jl"),
                        tspan=(0.0, 0.2),
                        l2=[
                            6.481654414510423e-05,
                            7.016772268337112e-05,
                            0.00012094672530945635,
                            8.88895396553818e-05,
                            9.519488369476238e-05,
                            4.8346561557994794e-05
                        ],
                        linf=[
                            0.0017873466830857,
                            0.0026647390422136507,
                            0.003916829320799509,
                            0.003792137288444195,
                            0.00292414430770023,
                            0.0020445616357903855
                        ])
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_fresnel.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_fresnel.jl"))
    errors = analysis_callback(sol)
    @test maximum(errors.l2[1:6]) < 2e-2
    @test all(iszero, errors.l2[7:9]) && all(iszero, errors.linf[7:9])
    # passive material components stay fixed
    u = Trixi.wrap_array(sol.u[end], semi)
    @test all(u_node -> u_node[7] in (1.0, 4.0) && u_node[8] == 1.0 && u_node[9] == 0.0,
              u)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_fresnel.jl (interface energy conservation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_fresnel.jl"),
                        surface_flux=FluxUpwindPenalty(0.0),
                        boundary_conditions=(;
                                             entire_boundary = boundary_condition_perfect_electric_conductor),
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

@trixi_testset "elixir_maxwell_3d_fresnel.jl (convergence)" begin
    using Trixi, TrixiMaxwell
    eocs, _ = Trixi.convergence_test(@__MODULE__,
                                     joinpath(EXAMPLES_DIR,
                                              "elixir_maxwell_3d_fresnel.jl"),
                                     3; polydeg = 2, cells_per_dimension = (8, 2, 2),
                                     cfl = 0.3)
    @test all(eocs[:l2][end, 1:6] .> 2.75)
end

@trixi_testset "elixir_maxwell_3d_dielectric_sphere.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_maxwell_3d_dielectric_sphere.jl"),
                        size_factor=2.0, tspan=(0.0, 1.0))
    u = Trixi.wrap_array(sol.u[end], semi)
    @test all(u_node -> u_node[7] in (1.0, 2.25) && u_node[8] == 1.0 && u_node[9] == 0.0,
              u)
    @test count(u_node -> u_node[7] == 2.25, u) > 0
    @test keys(mesh.boundary_faces) == (:SMA,)
    @test Trixi.integrate(energy_total, sol.u[end], semi) <
          Trixi.integrate(energy_total, sol.u[1], semi)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
@trixi_testset "elixir_maxwell_3d_mie.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_mie.jl"))
    @test mesh.md.mesh_type isa StartUpDG.CurvedMesh
    for (k, f) in enumerate(sigma.frequencies)
        f < 0.3 && continue
        mie = mie_efficiencies(sqrt(sphere_material.epsilon),
                               2 * pi * f * sphere_radius)
        @test sigma.scattering[k]≈mie.scattering * pi * sphere_radius^2 rtol=0.02
        @test abs(sigma.absorption[k]) < 0.03 * sigma.scattering[k]
    end
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
@trixi_testset "elixir_maxwell_3d_slab.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_slab.jl"))
    for (k, f) in enumerate(frequencies)
        airy = slab_transmittance_reflectance(sqrt(slab_material.epsilon),
                                              slab_thickness,
                                              f)
        @test transmittance[k]≈airy.transmittance atol=5e-3
        @test reflectance[k]≈airy.reflectance atol=1e-3
    end
    @test_throws ArgumentError DetectorPlaneCallback(semi, incident_field, frequencies;
                                                     point = (0.0, 0.0, -0.53),
                                                     normal = (0.0, 0.0, 1.0))
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
@trixi_testset "elixir_maxwell_3d_dipole.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_dipole.jl"))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@trixi_testset "elixir_maxwell_3d_tfsf.jl" begin
    # the pulse centre is inside the box at t = 1.5
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_tfsf.jl"),
                        tspan=(0.0, 1.5))
    mesh_, equations_, solver_, _ = Trixi.mesh_equations_solver_cache(semi)
    md = mesh_.md
    u = Trixi.wrap_array(sol.u[end], semi)
    energy_in = 0.0
    energy_out = 0.0
    for element in Base.OneTo(md.num_elements)
        energy = sum(solver_.basis.M *
                     energy_total.(view(u, :, element), Ref(equations_))) *
                 md.J[1, element]
        if is_total_field(TrixiMaxwell.element_centroid(md, element))
            energy_in += energy
        else
            energy_out += energy
        end
    end
    # exact energy of the pulse: cross-section 1 times the integral of exp(-2 (x / w)^2)
    @test energy_in≈0.25 * sqrt(pi / 2) rtol=5e-3
    @test energy_out < 1e-6 * energy_in
    peak_energy = Trixi.integrate(energy_total, sol.u[end], semi)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # the pulse has left the box through its far face and nothing remains
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_tfsf.jl"))
    @test Trixi.integrate(energy_total, sol.u[end], semi) < 1e-9 * peak_energy
end

@trixi_testset "elixir_maxwell_3d_tfsf.jl with CrossSectionCallback" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_tfsf.jl"),
                        callbacks=CallbackSet(summary_callback,
                                              CrossSectionCallback(semi, tfsf,
                                                                   [0.5, 1.0, 1.5]),
                                              stepsize_callback))
    cross_section_callback = callbacks.discrete_callbacks[2]
    sigma = cross_sections(cross_section_callback)
    @test sigma.frequencies == [0.5, 1.0, 1.5]
    # without a scatterer nothing is scattered or absorbed by the unit box face
    @test all(abs.(sigma.scattering) .< 1e-4)
    @test all(abs.(sigma.extinction) .< 5e-3)
    # transform of the Gaussian signal: w sqrt(pi) exp(-(pi f w)^2)
    incident = cross_section_callback.affect!.incident
    for (k, f) in enumerate(sigma.frequencies)
        spectrum = 0.25 * sqrt(pi) * exp(-(pi * f * 0.25)^2)
        @test sqrt(sum(abs2, view(incident, 1:3, 1, k)))≈spectrum rtol=1e-3
    end
end
@trixi_testset "elixir_maxwell_3d_pec_sphere.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_pec_sphere.jl"),
                        tspan=(0.0, 0.5))
    @test keys(mesh.boundary_faces) == (:sma, :tag_1)
    @test length(tfsf.faces) == 2 * 1470
    @test Trixi.integrate(energy_total, sol.u[end], semi) > 0
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
@trixi_testset "elixir_maxwell_3d_pml.jl" begin
    using LinearAlgebra: norm
    using StaticArrays: SVector
    probes = [SVector(0.5, 0.0, 0.0), SVector(0.0, 0.0, 0.5), SVector(0.3, 0.2, 0.4),
        SVector(-0.4, 0.1, -0.5)]
    fields(v) = SVector(v[1], v[2], v[3], v[4], v[5], v[6])
    function residual(sol, semi, dipole_field, equations)
        numerical = PointEvaluator(probes, semi)(sol.u[end], semi)
        exact = [dipole_field(x, sol.t[end], equations) for x in probes]
        return norm(reduce(vcat, fields.(numerical)) - reduce(vcat, fields.(exact)))
    end

    # coarser mesh with a two-cell layer; after the pulse has passed, only
    # reflections from the layer remain at the probes
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_pml.jl"),
                        cells_per_dimension=(8, 8, 8), pml_thickness=0.75)
    @test Trixi.nvariables(equations) == 12
    reflection_pml = residual(sol, semi, dipole_field, equations)
    u = Trixi.wrap_array(sol.u[end], semi)
    md = mesh.md
    # auxiliary fields vanish in elements that lie entirely in the physical region
    inside(element) = all(maximum(abs, view(md.xyz[d], :, element)) < 0.75 for d in 1:3)
    @test count(element -> inside(element) &&
                    any(u_node -> any(!iszero, u_node[7:12]),
                        view(u, :, element)), axes(u, 2)) == 0
    @test count(u_node -> any(!iszero, u_node[7:12]), u) > 0
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)

    # Silver-Mueller alone on the same domain reflects far more
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_maxwell_3d_pml.jl"),
                        cells_per_dimension=(8, 8, 8),
                        equations=MaxwellEquations3D(), source_terms=dipole)
    reflection_silver_mueller = residual(sol, semi, dipole_field, equations)
    @test reflection_pml < 0.2 * reflection_silver_mueller
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true, force = true)

end # module
