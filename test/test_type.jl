module TestType

using Test
using Trixi
using TrixiMaxwell
using StaticArrays: SVector

include("test_trixi.jl")

@testset "Test Type Stability" begin
    @timed_testset "Maxwell 3D" begin
        for RealT in (Float32, Float64)
            equations = @inferred MaxwellEquations3D(epsilon = one(RealT), mu = one(RealT))
            @test equations isa
                  MaxwellEquations3D{Homogeneous, NonDispersive, NoPML, 6, RealT}
            @test (@inferred similar(equations, RealT)) isa
                  MaxwellEquations3D{Homogeneous, NonDispersive, NoPML, 6, RealT}

            x = SVector(zero(RealT), zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = SVector(ntuple(_ -> one(RealT), 6))
            normal_direction = SVector(one(RealT), one(RealT), zero(RealT))

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT

            for orientation in 1:3
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_upwind(u_ll, u_rr, orientation, equations)) ==
                      RealT
                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
            end
            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_upwind(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred FluxUpwindPenalty(0.0)(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            @test eltype(@inferred flux_lax_friedrichs(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT

            @test typeof(@inferred permittivity(u, equations)) == RealT
            @test typeof(@inferred permeability(u, equations)) == RealT
            @test typeof(@inferred conductivity(u, equations)) == RealT
            @test typeof(@inferred impedance(u, equations)) == RealT
            @test typeof(@inferred admittance(u, equations)) == RealT
            @test typeof(@inferred speed_of_light(u, equations)) == RealT
            @test eltype(@inferred source_terms_conductivity(u, x, t, equations)) == RealT

            for boundary_condition in (boundary_condition_perfect_electric_conductor,
                                       boundary_condition_perfect_magnetic_conductor,
                                       boundary_condition_silver_mueller,
                                       BoundaryConditionIncidentField(initial_condition_convergence_test))
                @test eltype(@inferred boundary_condition(u, normal_direction, x, t,
                                                          flux_upwind, equations)) == RealT
                for direction in (1, 2)
                    @test eltype(@inferred boundary_condition(u, normal_direction,
                                                              direction, x, t,
                                                              flux_upwind, equations)) ==
                          RealT
                    @test eltype(@inferred boundary_condition(u, 1, direction, x, t,
                                                              flux_upwind, equations)) ==
                          RealT
                end
            end

            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test typeof(@inferred energy_total(u, equations)) == RealT
        end
    end
    @timed_testset "Maxwell 3D heterogeneous" begin
        for RealT in (Float32, Float64)
            equations = @inferred MaxwellEquations3D(Heterogeneous(); epsilon = one(RealT),
                                                     mu = one(RealT), sigma = zero(RealT))
            @test equations isa
                  MaxwellEquations3D{Heterogeneous, NonDispersive, NoPML, 9, RealT}

            x = SVector(zero(RealT), zero(RealT), zero(RealT))
            t = zero(RealT)
            u_ll = SVector(ntuple(i -> i <= 6 ? one(RealT) : RealT(1 + i), 9))
            u_rr = SVector(ntuple(i -> i <= 6 ? RealT(2) : RealT(2 + i), 9))
            normal_direction = SVector(one(RealT), one(RealT), zero(RealT))

            for orientation in 1:3
                @test eltype(@inferred flux(u_ll, orientation, equations)) == RealT
                @test eltype(@inferred flux_upwind(u_ll, u_rr, orientation, equations)) ==
                      RealT
            end
            @test eltype(@inferred flux(u_ll, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_upwind(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred FluxUpwindPenalty(0.0)(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u_ll, equations)) == RealT
            @test typeof(@inferred permittivity(u_ll, equations)) == RealT
            @test typeof(@inferred impedance(u_ll, equations)) == RealT
            @test typeof(@inferred admittance(u_ll, equations)) == RealT
            @test typeof(@inferred speed_of_light(u_ll, equations)) == RealT
            @test eltype(@inferred cons2entropy(u_ll, equations)) == RealT
            @test typeof(@inferred energy_total(u_ll, equations)) == RealT
            @test eltype(@inferred source_terms_conductivity(u_ll, x, t, equations)) ==
                  RealT
            @test eltype(@inferred TrixiMaxwell.with_passive_defaults(SVector(ntuple(_ -> one(RealT),
                                                                                     6)),
                                                                      equations)) == RealT
            for boundary_condition in (boundary_condition_perfect_electric_conductor,
                                       boundary_condition_perfect_magnetic_conductor,
                                       boundary_condition_silver_mueller)
                @test eltype(@inferred boundary_condition(u_ll, normal_direction, x, t,
                                                          flux_upwind, equations)) == RealT
            end
        end
    end
    @timed_testset "Sources and incident fields" begin
        for RealT in (Float32, Float64)
            equations = MaxwellEquations3D(epsilon = one(RealT), mu = one(RealT))
            heterogeneous = MaxwellEquations3D(Heterogeneous(); epsilon = one(RealT),
                                               mu = one(RealT))
            x = SVector(RealT(0.3), RealT(0.2), RealT(0.1))
            t = RealT(0.5)
            signal = GaussianPulse(RealT(0.3); delay = RealT(1))
            modulated = ModulatedGaussianPulse(RealT(2), RealT(0.3))
            for s in (signal, modulated)
                @test typeof(@inferred s(t)) == RealT
                @test typeof(@inferred signal_derivative(s, t)) == RealT
                @test typeof(@inferred signal_second_derivative(s, t)) == RealT
            end

            wave = PlaneWave(SVector(one(RealT), zero(RealT), zero(RealT)),
                             SVector(zero(RealT), zero(RealT), one(RealT)), signal)
            @test eltype(@inferred wave(x, t, equations)) == RealT
            @test eltype(@inferred wave(x, t, heterogeneous)) == RealT
            @test eltype(@inferred (wave + wave)(x, t, equations)) == RealT
            @test eltype(@inferred initial_condition_zero(x, t, heterogeneous)) == RealT

            dipole = HertzianDipole(zero(x), SVector(zero(RealT), zero(RealT), one(RealT)),
                                    RealT(0.1), signal)
            u = SVector(ntuple(_ -> one(RealT), 6))
            @test eltype(@inferred dipole(u, x, t, equations)) == RealT
            @test eltype(@inferred dipole(vcat(u,
                                               SVector(one(RealT), one(RealT), zero(RealT))),
                                          x, t, heterogeneous)) == RealT
            field = @inferred HertzianDipoleField(dipole, equations)
            @test field isa HertzianDipoleField{RealT}
            @test eltype(@inferred field(x, t, equations)) == RealT
        end
    end
    @timed_testset "Uniaxial PML" begin
        for RealT in (Float32, Float64)
            equations = @inferred MaxwellEquations3D(UPML(); epsilon = one(RealT))
            @test equations isa
                  MaxwellEquations3D{Homogeneous, NonDispersive, UPML, 12, RealT}
            @test (@inferred similar(equations, RealT)) isa
                  MaxwellEquations3D{Homogeneous, NonDispersive, UPML, 12, RealT}
            x = SVector(RealT(1.25), zero(RealT), zero(RealT))
            t = zero(RealT)
            u = SVector(ntuple(_ -> one(RealT), 12))
            profile = PMLProfile(SVector(-RealT(1.5), -RealT(1.5), -RealT(1.5)),
                                 SVector(RealT(1.5), RealT(1.5), RealT(1.5)), RealT(0.5))
            @test eltype(@inferred profile(x)) == RealT
            source = SourceTermsPML(profile)
            @test eltype(@inferred source(u, x, t, equations)) == RealT
            combined = CombinedSourceTerms(source, source_terms_conductivity)
            @test eltype(@inferred combined(u, x, t, equations)) == RealT
            @test eltype(@inferred flux(u, x, equations)) == RealT
            @test eltype(@inferred flux_upwind(u, u, x, equations)) == RealT
            @test eltype(@inferred TrixiMaxwell.with_passive_defaults(SVector(ntuple(_ -> one(RealT),
                                                                                     6)),
                                                                      equations)) == RealT
            @test eltype(@inferred initial_condition_zero(x, t, equations)) == RealT
        end
    end
end

end # module
