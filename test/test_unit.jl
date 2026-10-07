module TestUnit

using Test
using Trixi
using TrixiMaxwell
using StaticArrays: SVector
using LinearAlgebra: norm, dot, cross

include("test_trixi.jl")

# constant fields with placeholder materials, for the set_materials! test
function initial_condition_fresnel_like(x, t, equations::MaxwellEquations3D{Heterogeneous})
    return SVector(0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
end

@testset "Unit tests" begin
#! format: noindent

@timed_testset "MaxwellEquations3D" begin
    equations = MaxwellEquations3D()

    @test equations isa MaxwellEquations3D{Homogeneous, NoPML, 6, Float64}
    @test equations isa Trixi.AbstractMaxwellEquations{3, 6}
    @test ndims(equations) == 3
    @test Trixi.nvariables(equations) == 6
    @test equations.epsilon == 1.0
    @test equations.mu == 1.0

    equations = MaxwellEquations3D(epsilon = 4, mu = 1.0)
    @test equations isa MaxwellEquations3D{Homogeneous, NoPML, 6, Float64}
    @test permittivity(equations) == 4.0
    @test permeability(equations) == 1.0
    @test conductivity(equations) == 0.0
    @test impedance(equations) == 0.5
    @test admittance(equations) == 2.0
    @test speed_of_light(equations) == 0.5
    @test MaxwellEquations3D(epsilon = 4.0, sigma = 0.25).sigma == 0.25
    @test MaxwellEquations3D(epsilon = 4.0f0) isa
          MaxwellEquations3D{Homogeneous, NoPML, 6, Float32}

    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    @test permittivity(u, equations) == 4.0
    @test permeability(u, equations) == 1.0
    @test impedance(u, equations) == 0.5
    @test speed_of_light(u, equations) == 0.5

    equations32 = similar(equations, Float32)
    @test equations32 isa MaxwellEquations3D{Homogeneous, NoPML, 6, Float32}
    @test equations32.epsilon == 4.0f0

    expected_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    @test Trixi.varnames(Trixi.cons2cons, equations) == expected_names
    @test Trixi.varnames(Trixi.cons2prim, equations) == expected_names

    @test Trixi.have_constant_speed(equations) === Trixi.True()
end

@timed_testset "Physical flux" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test flux(u, 1, equations) == SVector(0.0, 1.5, -1.25, 0.0, -3.0, 2.0)
    @test flux(u, 2, equations) == SVector(-1.5, 0.0, 1.0, 3.0, 0.0, -1.0)
    @test flux(u, 3, equations) == SVector(1.25, -1.0, 0.0, -2.0, 1.0, 0.0)

    for orientation in 1:3
        normal = SVector(ntuple(i -> i == orientation ? 1.0 : 0.0, 3))
        @test flux(u, normal, equations) == flux(u, orientation, equations)
    end

    normal = SVector(2.0, -1.0, 0.5)
    @test flux(u, normal, equations) == SVector(2.125, 2.5, -3.5, -4.0, -5.5, 5.0)

    # linear in the normal direction
    @test flux(u, 3 * normal, equations) ≈ 3 * flux(u, normal, equations)
end

@timed_testset "Wave speeds" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = -u_ll

    for orientation in 1:3
        @test max_abs_speed_naive(u_ll, u_rr, orientation, equations) == 0.5
    end

    # norm(normal) = 3
    normal = SVector(2.0, -1.0, 2.0)
    @test max_abs_speed_naive(u_ll, u_rr, normal, equations) == 1.5
    @test Trixi.max_abs_speed(u_ll, u_rr, normal, equations) == 1.5

    @test Trixi.max_abs_speeds(equations) == (0.5, 0.5, 0.5)
    @test Trixi.max_abs_speeds(u_ll, equations) == (0.5, 0.5, 0.5)
end

@timed_testset "Upwind flux" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    c = speed_of_light(equations)
    flux_central_penalty = FluxUpwindPenalty(0.0)

    u_ll = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u_rr = SVector(-2.0, 0.5, 4.0, -1.0, 3.0, -0.5)
    normal = SVector(2.0, -1.0, 2.0)
    n_hat = normal / norm(normal)

    @test flux_upwind isa FluxUpwindPenalty
    @test flux_upwind.alpha == 1.0

    # consistency
    @test flux_upwind(u_ll, u_ll, normal, equations) == flux(u_ll, normal, equations)
    for orientation in 1:3
        @test flux_upwind(u_ll, u_ll, orientation, equations) ==
              flux(u_ll, orientation, equations)
        @test flux_upwind(u_ll, u_rr, orientation, equations) ==
              flux_upwind(u_ll, u_rr,
                          SVector(ntuple(i -> i == orientation ? 1.0 : 0.0, 3)),
                          equations)
    end

    # symmetry
    @test flux_upwind(u_ll, u_rr, normal, equations) ≈
          -flux_upwind(u_rr, u_ll, -normal, equations)

    # homogeneous of degree one in the normal direction
    @test flux_upwind(u_ll, u_rr, 3 * normal, equations) ≈
          3 * flux_upwind(u_ll, u_rr, normal, equations)

    # alpha = 0 is the central flux
    @test flux_central_penalty(u_ll, u_rr, normal, equations) ≈
          0.5 * (flux(u_ll, normal, equations) + flux(u_rr, normal, equations))

    # jumps in the normal components carry no penalty
    normal_jump = vcat(0.7 * n_hat, -1.3 * n_hat)
    @test flux_upwind(u_ll, u_ll + normal_jump, normal, equations) ≈
          flux_central_penalty(u_ll, u_ll + normal_jump, normal, equations)

    # tangential jumps are penalized with speed c, like Lax-Friedrichs
    tangent = cross(n_hat, SVector(0.0, 0.0, 1.0))
    tangential_jump = vcat(0.7 * tangent, -1.3 * tangent)
    @test flux_upwind(u_ll, u_ll + tangential_jump, normal, equations) ≈
          flux_lax_friedrichs(u_ll, u_ll + tangential_jump, normal, equations)

    # general jump: penalty acts on the tangential projection only
    dE = SVector(u_rr[1] - u_ll[1], u_rr[2] - u_ll[2], u_rr[3] - u_ll[3])
    dH = SVector(u_rr[4] - u_ll[4], u_rr[5] - u_ll[5], u_rr[6] - u_ll[6])
    dE_t = dE - dot(dE, n_hat) * n_hat
    dH_t = dH - dot(dH, n_hat) * n_hat
    @test flux_upwind(u_ll, u_rr, normal, equations) -
          flux_central_penalty(u_ll, u_rr, normal, equations) ≈
          -0.5 * c * norm(normal) * vcat(dE_t, dH_t)

    # alpha scales the penalty linearly
    @test FluxUpwindPenalty(0.5)(u_ll, u_rr, normal, equations) ≈
          0.5 * (flux_upwind(u_ll, u_rr, normal, equations) +
           flux_central_penalty(u_ll, u_rr, normal, equations))

    @test sprint(show, FluxUpwindPenalty(0.25)) == "FluxUpwindPenalty(alpha=0.25)"
end

@timed_testset "Boundary conditions" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u_inner = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    normal = SVector(2.0, -1.0, 2.0) / 3
    x = SVector(0.1, 0.2, 0.3)
    t = 0.4
    flux_central_penalty = FluxUpwindPenalty(0.0)

    u_pec = SVector(-1.0, -2.0, -3.0, 4.0, 5.0, 6.0)
    u_pmc = SVector(1.0, 2.0, 3.0, -4.0, -5.0, -6.0)
    incident = BoundaryConditionIncidentField(initial_condition_convergence_test)
    dirichlet = BoundaryConditionDirichlet(initial_condition_convergence_test)

    for surface_flux in (flux_upwind, flux_central_penalty)
        @test boundary_condition_perfect_electric_conductor(u_inner, normal, x, t,
                                                            surface_flux, equations) ==
              surface_flux(u_inner, u_pec, normal, equations)
        @test boundary_condition_perfect_magnetic_conductor(u_inner, normal, x, t,
                                                            surface_flux, equations) ==
              surface_flux(u_inner, u_pmc, normal, equations)
        # absorbing flux does not depend on the surface flux of the scheme
        @test boundary_condition_silver_mueller(u_inner, normal, x, t, surface_flux,
                                                equations) ==
              flux_upwind(u_inner, zero(u_inner), normal, equations)
        @test incident(u_inner, normal, x, t, surface_flux, equations) ==
              dirichlet(u_inner, normal, x, t, surface_flux, equations)
    end

    # TreeMesh and StructuredMesh signatures follow Trixi's Dirichlet convention
    for direction in 1:6, surface_flux in (flux_upwind, flux_central_penalty)
        orientation = (direction + 1) ÷ 2
        unit_normal = SVector(ntuple(i -> i == orientation ? 1.0 : 0.0, 3))
        @test incident(u_inner, orientation, direction, x, t, surface_flux,
                       equations) ≈
              dirichlet(u_inner, orientation, direction, x, t, surface_flux, equations)
        @test incident(u_inner, unit_normal, direction, x, t, surface_flux,
                       equations) ≈
              dirichlet(u_inner, unit_normal, direction, x, t, surface_flux, equations)
        @test boundary_condition_perfect_electric_conductor(u_inner, orientation,
                                                            direction, x, t,
                                                            surface_flux, equations) ==
              boundary_condition_perfect_electric_conductor(u_inner, unit_normal,
                                                            direction, x, t,
                                                            surface_flux, equations)
    end

    # mirrored tangential fields cancel in the central average
    f_pec = boundary_condition_perfect_electric_conductor(u_inner, normal, x, t,
                                                          flux_central_penalty,
                                                          equations)
    @test all(iszero, f_pec[4:6])
    f_pmc = boundary_condition_perfect_magnetic_conductor(u_inner, normal, x, t,
                                                          flux_central_penalty,
                                                          equations)
    @test all(iszero, f_pmc[1:3])

    # an outgoing plane wave passes the absorbing boundary with the physical flux
    normal_x = SVector(1.0, 0.0, 0.0)
    u_outgoing = initial_condition_convergence_test(x, 0.0, equations)
    @test boundary_condition_silver_mueller(u_outgoing, normal_x, x, t, flux_upwind,
                                            equations) ≈
          flux(u_outgoing, normal_x, equations)

    # energy only leaves through the absorbing boundary
    f_sm = boundary_condition_silver_mueller(u_inner, normal, x, t, flux_upwind,
                                             equations)
    @test dot(cons2entropy(u_inner, equations), f_sm) >= 0
end

@timed_testset "ImportedMesh" begin
    dg = DGMulti(polydeg = 1, element_type = Tet(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_upwind),
                 volume_integral = VolumeIntegralWeakForm())
    tol = 1e-12
    predicates = (; x_min = x -> abs(x[1] + 1) < tol, x_max = x -> abs(x[1] - 1) < tol,
                  y_min = x -> abs(x[2] + 1) < tol, y_max = x -> abs(x[2] - 1) < tol,
                  z_min = x -> abs(x[3] + 1) < tol, z_max = x -> abs(x[3] - 1) < tol)
    reference = DGMultiMesh(dg, (2, 2, 2); coordinates_min = (-1.0, -1.0, -1.0),
                            coordinates_max = (1.0, 1.0, 1.0),
                            is_on_boundary = predicates)
    VX, VY, VZ = reference.md.mesh_type.VXYZ
    EToV = reference.md.mesh_type.EToV

    # tag faces by evaluating the same predicates on vertex coordinates
    face_sets = Dict{Int, Vector{NTuple{3, Int}}}()
    names = Dict{Int, String}()
    for (tag, (key, predicate)) in enumerate(pairs(predicates))
        names[tag] = String(key)
        faces = NTuple{3, Int}[]
        for e in axes(EToV, 1), vertices in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
            ids = Tuple(EToV[e, v] for v in vertices)
            if all(i -> predicate((VX[i], VY[i], VZ[i])), ids)
                push!(faces, ids)
            end
        end
        face_sets[tag] = faces
    end

    imported = ImportedMesh((VX, VY, VZ), EToV; face_sets, face_set_names = names)
    @test imported isa ImportedMesh{Float64}
    @test TrixiMaxwell.num_elements(imported) == size(EToV, 1)
    @test all(==(1), imported.element_groups)
    @test occursin("48 tetrahedra", sprint(show, imported))

    mesh = DGMultiMesh(dg, imported)
    @test sort(collect(keys(mesh.boundary_faces))) ==
          sort(collect(keys(reference.boundary_faces)))
    for key in keys(mesh.boundary_faces)
        @test mesh.boundary_faces[key] == sort(reference.boundary_faces[key])
    end

    # reorientation does not change the face sets
    flipped = copy(EToV)
    for e in 1:3:size(flipped, 1)
        flipped[e, 1], flipped[e, 2] = flipped[e, 2], flipped[e, 1]
    end
    imported_flipped = ImportedMesh((VX, VY, VZ), flipped; face_sets,
                                    face_set_names = names)
    mesh_flipped = DGMultiMesh(dg, imported_flipped)
    @test mesh_flipped.md.J ≈ mesh.md.J
    for key in keys(mesh.boundary_faces)
        @test length(mesh_flipped.boundary_faces[key]) ==
              length(mesh.boundary_faces[key])
    end

    # predicates merged with file tags
    partial = Dict(tag => face_sets[tag] for tag in 1:5)
    partial_names = Dict(tag => names[tag] for tag in 1:5)
    imported_partial = ImportedMesh((VX, VY, VZ), EToV; face_sets = partial,
                                    face_set_names = partial_names)
    @test_throws ArgumentError DGMultiMesh(dg, imported_partial)
    merged = DGMultiMesh(dg, imported_partial;
                         is_on_boundary = (; z_max = predicates.z_max))
    @test merged.boundary_faces.z_max == sort(reference.boundary_faces.z_max)
    @test_throws ArgumentError DGMultiMesh(dg, imported_partial;
                                           is_on_boundary = (;
                                                             x_min = predicates.x_min))
    @test_throws ArgumentError DGMultiMesh(dg, imported;
                                           is_on_boundary = (; top = predicates.z_max))
    relaxed = DGMultiMesh(dg, imported_partial; allow_untagged_boundary = true)
    @test length(relaxed.boundary_faces) == 5

    # error paths
    fv = dg.basis.fv
    interior_id = findfirst(i -> reference.md.FToF[i] != i,
                            eachindex(reference.md.FToF))
    element, face = (interior_id - 1) ÷ 4 + 1, (interior_id - 1) % 4 + 1
    interior = Tuple(EToV[element, v] for v in fv[face])
    boundary = first(face_sets[1])

    # interior-only sets are skipped, mixed sets are rejected
    extra_set = copy(face_sets)
    extra_set[7] = [interior]
    skipped = DGMultiMesh(dg,
                          ImportedMesh((VX, VY, VZ), EToV; face_sets = extra_set,
                                       face_set_names = names))
    @test length(skipped.boundary_faces) == 6
    extra_set[7] = [boundary, interior]
    @test_throws ArgumentError DGMultiMesh(dg,
                                           ImportedMesh((VX, VY, VZ), EToV;
                                                        face_sets = extra_set))
    # a triple that is no face
    extra_set[7] = [(1, 1, 2)]
    @test_throws ArgumentError DGMultiMesh(dg,
                                           ImportedMesh((VX, VY, VZ), EToV;
                                                        face_sets = extra_set))
    # a boundary face in two sets
    extra_set[7] = [boundary]
    @test_throws ArgumentError DGMultiMesh(dg,
                                           ImportedMesh((VX, VY, VZ), EToV;
                                                        face_sets = extra_set))

    @test_throws ArgumentError ImportedMesh((VX, VY, VZ), EToV[:, 1:3])
    @test_throws ArgumentError ImportedMesh((VX, VY, VZ[1:(end - 1)]), EToV)
    @test_throws ArgumentError ImportedMesh((VX, VY, VZ), EToV; element_groups = [1])
    @test_throws ArgumentError ImportedMesh((VX, VY, VZ), EToV;
                                            group_names = Dict(2 => "x"))
    @test_throws ArgumentError ImportedMesh((VX, VY, VZ), EToV;
                                            face_set_names = Dict(1 => "x"))
    too_large = copy(EToV)
    too_large[1, 1] = length(VX) + 1
    @test_throws ArgumentError ImportedMesh((VX, VY, VZ), too_large)

    # VTK export of the imported mesh
    vtk = TrixiMaxwell.mesh_vtk_data(imported)
    @test size(vtk.points) == (3, length(VX))
    @test length(vtk.volume_cells) == size(EToV, 1)
    @test length(vtk.face_cells) == sum(length, values(face_sets))
    @test all(==(1), vtk.face_on_boundary)
    for tag in 1:6
        @test count(==(tag), vtk.face_tags) == length(face_sets[tag])
    end
    vtk_mixed = TrixiMaxwell.mesh_vtk_data(ImportedMesh((VX, VY, VZ), EToV;
                                                        face_sets = Dict(1 => [interior])))
    @test vtk_mixed.face_on_boundary == [0]
    mktempdir() do dir
        files = write_mesh_vtk(imported, joinpath(dir, "cube"))
        @test basename.(files) == ["cube_volume.vtu", "cube_faces.vtu"]
        @test all(isfile, files)
        files = write_mesh_vtk(ImportedMesh((VX, VY, VZ), EToV), joinpath(dir, "bare"))
        @test basename.(files) == ["bare_volume.vtu"]
    end

    imported32 = ImportedMesh(map(v -> Float32.(v), (VX, VY, VZ)), EToV; face_sets,
                              face_set_names = names)
    @test imported32 isa ImportedMesh{Float32}

    @test TrixiMaxwell.face_set_key(3, Dict{Int, String}()) == :tag_3
    @test TrixiMaxwell.face_set_key(3, Dict(3 => "Wall")) == :Wall
end

@timed_testset "Heterogeneous materials" begin
    equations = MaxwellEquations3D(Heterogeneous(); epsilon = 4.0, mu = 1.0,
                                   sigma = 0.5)
    @test equations isa MaxwellEquations3D{Heterogeneous, NoPML, 9, Float64}
    @test Trixi.nvariables(equations) == 9
    @test Trixi.varnames(Trixi.cons2cons, equations)[7:9] == ("epsilon", "mu", "sigma")
    @test Trixi.varnames(Trixi.cons2prim, equations) ==
          Trixi.varnames(Trixi.cons2cons, equations)
    @test Trixi.have_constant_speed(equations) === Trixi.False()
    @test similar(equations, Float32) isa
          MaxwellEquations3D{Heterogeneous, NoPML, 9, Float32}

    fields = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u = vcat(fields, SVector(4.0, 1.0, 0.5))
    @test permittivity(u, equations) == u[7]
    @test permeability(u, equations) == u[8]
    @test conductivity(u, equations) == u[9]
    @test impedance(u, equations) == 0.5
    @test admittance(u, equations) == 2.0
    @test speed_of_light(u, equations) == 0.5
    @test Trixi.max_abs_speeds(u, equations) == (0.5, 0.5, 0.5)
    @test_throws MethodError permittivity(equations)
    @test_throws MethodError initial_condition_convergence_test(SVector(0.0, 0.0, 0.0),
                                                                0.0,
                                                                equations)

    # same material on both sides: identical to the homogeneous flux, passive slots zero
    homogeneous = MaxwellEquations3D(epsilon = 4.0, mu = 1.0, sigma = 0.5)
    other = SVector(-2.0, 0.5, 4.0, -1.0, 3.0, -0.5)
    u_rr = vcat(other, SVector(4.0, 1.0, 0.5))
    normal = SVector(2.0, -1.0, 2.0)
    for orientation_or_normal in (1, 2, 3, normal)
        @test flux(u, orientation_or_normal, equations)[1:6] ==
              flux(fields, orientation_or_normal, homogeneous)
        @test all(iszero, flux(u, orientation_or_normal, equations)[7:9])
        @test flux_upwind(u, u_rr, orientation_or_normal, equations)[1:6] ≈
              flux_upwind(fields, other, orientation_or_normal, homogeneous)
        @test all(iszero, flux_upwind(u, u_rr, orientation_or_normal, equations)[7:9])
    end
    @test flux_upwind(u, u, normal, equations) == flux(u, normal, equations)
    @test cons2entropy(u, equations) == vcat(4.0 * fields[1:3], fields[4:6], zeros(3))
    @test energy_total(u, equations) == energy_total(fields, homogeneous)

    # different materials: passive slots stay exact zeros, each side uses its own material
    u_glass = vcat(other, SVector(2.25, 1.0, 0.0))
    f_ll = flux_upwind(u, u_glass, normal, equations)
    f_rr = flux_upwind(u_glass, u, -normal, equations)
    @test all(iszero, f_ll[7:9]) && all(iszero, f_rr[7:9])
    @test f_ll[1:6] != -f_rr[1:6]   # not conservative across the interface

    # interface balance: continuous tangential fields give the physical flux on both sides
    n_hat = normal / norm(normal)
    E_ll = SVector(1.0, 2.0, 3.0)
    H_ll = SVector(4.0, 5.0, 6.0)
    eps_ll, eps_rr = 4.0, 2.25
    E_rr = E_ll + (eps_ll / eps_rr - 1) * dot(E_ll, n_hat) * n_hat   # normal D continuous
    u_bal_ll = vcat(E_ll, H_ll, SVector(eps_ll, 1.0, 0.0))
    u_bal_rr = vcat(E_rr, H_ll, SVector(eps_rr, 1.0, 0.0))
    @test flux_upwind(u_bal_ll, u_bal_rr, normal, equations) ≈
          flux(u_bal_ll, normal, equations)
    @test flux_upwind(u_bal_rr, u_bal_ll, -normal, equations) ≈
          flux(u_bal_rr, -normal, equations)

    # boundary conditions keep the interior material
    x = SVector(0.1, 0.2, 0.3)
    for bc in (boundary_condition_perfect_electric_conductor,
               boundary_condition_perfect_magnetic_conductor,
               boundary_condition_silver_mueller,
               BoundaryConditionIncidentField((x, t, eq) -> vcat(other,
                                                                 SVector(1.0, 1.0, 0.0))))
        f = bc(u, n_hat, x, 0.0, flux_upwind, equations)
        @test length(f) == 9 && all(iszero, f[7:9])
    end
    @test boundary_condition_perfect_electric_conductor(u, n_hat, x, 0.0, flux_upwind,
                                                        equations)[1:6] ==
          boundary_condition_perfect_electric_conductor(fields, n_hat, x, 0.0,
                                                        flux_upwind,
                                                        homogeneous)
    @test boundary_condition_silver_mueller(u, n_hat, x, 0.0, flux_upwind, equations)[1:6] ==
          boundary_condition_silver_mueller(fields, n_hat, x, 0.0, flux_upwind,
                                            homogeneous)

    # conductivity source: -sigma E / epsilon on E only
    source = source_terms_conductivity(u, x, 0.0, equations)
    @test source[1:3] == -0.125 * fields[1:3]
    @test all(iszero, source[4:9])
    @test source_terms_conductivity(fields, x, 0.0, homogeneous) ==
          vcat(-0.125 * fields[1:3], zeros(3))
    @test all(iszero,
              source_terms_conductivity(fields, x, 0.0,
                                        MaxwellEquations3D(epsilon = 4.0)))

    # analytic initial conditions are homogeneous; heterogeneous elixirs define their own
    @test_throws MethodError initial_condition_cavity(x, 0.0, equations)
    @test TrixiMaxwell.with_passive_defaults(fields, equations) ==
          vcat(fields, [4.0, 1.0, 0.5])
    @test TrixiMaxwell.with_passive_defaults(fields, homogeneous) === fields
end

@timed_testset "Material and set_materials!" begin
    @test Material() == Material(1.0, 1.0, 0.0)
    @test Material(epsilon = 4) isa Material{Float64}
    @test Material(epsilon = 4.0f0).mu === 1.0f0
    @test TrixiMaxwell.material_components(Material(epsilon = 4.0, sigma = 0.5)) ==
          SVector(4.0, 1.0, 0.5)

    dg = DGMulti(polydeg = 2, element_type = Tet(), approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_upwind),
                 volume_integral = VolumeIntegralWeakForm())
    mesh = DGMultiMesh(dg, (4, 2, 2); coordinates_min = (-1.0, 0.0, 0.0),
                       coordinates_max = (1.0, 1.0, 1.0))
    equations = MaxwellEquations3D(Heterogeneous())
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_fresnel_like,
                                        dg;
                                        boundary_conditions = (;
                                                               entire_boundary = boundary_condition_perfect_electric_conductor))
    ode = semidiscretize(semi, (0.0, 1.0))

    glass = Material(epsilon = 2.25)
    set_materials!(ode.u0, semi, x -> x[1] < 0 ? Material() : glass)
    u = Trixi.wrap_array(ode.u0, semi)
    centroids = [TrixiMaxwell.element_centroid(mesh.md, e) for e in axes(u, 2)]
    expected(e) = centroids[e][1] < 0 ? SVector(1.0, 1.0, 0.0) : SVector(2.25, 1.0, 0.0)
    @test all(u[node, e][7:9] == expected(e) for e in axes(u, 2), node in axes(u, 1))
    fields0 = initial_condition_fresnel_like(SVector(0.0, 0.0, 0.0), 0.0, equations)[1:6]
    @test all(u[node, e][1:6] ≈ fields0 for e in axes(u, 2), node in axes(u, 1))
    @test count(c -> c[1] < 0, centroids) == length(centroids) ÷ 2

    groups = [c[1] < 0 ? 1 : 2 for c in centroids]
    set_materials!(ode.u0, semi, groups, Dict(1 => Material(sigma = 0.5), 2 => glass))
    @test all(u[1, e][7:9] ==
              (groups[e] == 1 ? SVector(1.0, 1.0, 0.5) : SVector(2.25, 1.0, 0.0))
              for e in axes(u, 2))
    @test_throws ArgumentError set_materials!(ode.u0, semi, groups, Dict(1 => glass))
    @test_throws ArgumentError set_materials!(ode.u0, semi, groups[1:3],
                                              Dict(1 => glass, 2 => glass))

    semi_homogeneous = SemidiscretizationHyperbolic(mesh, MaxwellEquations3D(),
                                                    initial_condition_cavity, dg;
                                                    boundary_conditions = (;
                                                                           entire_boundary = boundary_condition_perfect_electric_conductor))
    ode_homogeneous = semidiscretize(semi_homogeneous, (0.0, 1.0))
    @test_throws ArgumentError set_materials!(ode_homogeneous.u0, semi_homogeneous,
                                              x -> glass)
end

@timed_testset "Energy" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    @test cons2prim(u, equations) == u
    @test cons2entropy(u, equations) == SVector(4.0, 8.0, 12.0, 4.0, 5.0, 6.0)
    @test energy_total(u, equations) == 66.5
    @test dot(cons2entropy(u, equations), u) ≈ 2 * energy_total(u, equations)
end

@timed_testset "Plane-wave initial condition" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    c = speed_of_light(equations)
    Z = impedance(equations)
    initial_condition = initial_condition_convergence_test

    # quarter period: sin(2 pi x) = 1
    u_peak = initial_condition(SVector(0.25, 0.0, 0.0), 0.0, equations)
    @test u_peak ≈ SVector(0.0, 1.0, 0.0, 0.0, 0.0, 2.0)

    x = SVector(1 / 8, 1 / 3, 2 / 5)
    t = 0.125
    u = initial_condition(x, t, equations)
    x_shifted = SVector(x[1] - c * t, x[2], x[3])
    @test u ≈ initial_condition(x_shifted, 0.0, equations)

    E = SVector(u[1], u[2], u[3])
    H = SVector(u[4], u[5], u[6])
    @test norm(E) ≈ Z * norm(H)
    @test cross(E, H)[1] > 0
    @test cross(E, H)[2] == 0
    @test cross(E, H)[3] == 0

    # unit period in x
    @test initial_condition(SVector(0.3, 0.0, 0.0), 0.0, equations) ≈
          initial_condition(SVector(1.3, 0.0, 0.0), 0.0, equations)
end
@timed_testset "Time signals" begin
    h = 1.0e-4
    for signal in (GaussianPulse(0.3; delay = 1.0),
                   ModulatedGaussianPulse(2.0, 0.3; delay = 1.0, phase = 0.3))
        for t in (0.6, 0.87, 1.3)
            @test signal_derivative(signal, t)≈(signal(t + h) - signal(t - h)) / (2h) atol=1e-5
            @test signal_second_derivative(signal,
                                           t)≈
            (signal(t + h) - 2 * signal(t) + signal(t - h)) / h^2 atol=1e-3
        end
    end
    @test GaussianPulse(0.3; delay = 1.0)(1.0) == 1.0
    @test GaussianPulse(0.5f0) isa GaussianPulse{Float32}
    @test GaussianPulse(0.5; delay = 1) isa GaussianPulse{Float64}
    @test ModulatedGaussianPulse(2.0f0, 0.5f0) isa ModulatedGaussianPulse{Float32}
    @test ModulatedGaussianPulse(2.0, 0.5; phase = pi / 2)(0.0) == 1.0
end

@timed_testset "PlaneWave" begin
    equations = MaxwellEquations3D(epsilon = 4.0, mu = 1.0)
    c = speed_of_light(equations)
    Y = admittance(equations)
    signal = GaussianPulse(0.3; delay = 1.0)
    wave = PlaneWave((1.0, 1.0, 0.0), (0.0, 0.0, 2.0), signal)
    k = SVector(1.0, 1.0, 0.0) / sqrt(2)
    @test wave.direction ≈ k
    @test_throws ArgumentError PlaneWave((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), signal)

    x = SVector(0.3, -0.2, 0.7)
    t = 0.8
    u = wave(x, t, equations)
    E = SVector(u[1], u[2], u[3])
    H = SVector(u[4], u[5], u[6])
    @test E ≈ signal(t - dot(k, x) / c) * SVector(0.0, 0.0, 2.0)
    @test H ≈ Y * cross(k, E)
    dt = 0.17
    @test wave(x, t, equations) ≈ wave(x - c * dt * k, t - dt, equations)

    heterogeneous = MaxwellEquations3D(Heterogeneous(); epsilon = 4.0, mu = 1.0)
    u_het = wave(x, t, heterogeneous)
    @test length(u_het) == 9
    @test u_het[1:6] ≈ u
    @test u_het[7:9] == SVector(4.0, 1.0, 0.0)

    # circular polarization: two modulated pulses in quadrature keep |E| on the envelope
    frequency = 3.0
    width = 0.5
    circular = PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                         ModulatedGaussianPulse(frequency, width)) +
               PlaneWave((0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                         ModulatedGaussianPulse(frequency, width; phase = pi / 2))
    for t in (0.0, 0.1, 0.3)
        u = circular(SVector(0.0, 0.0, 0.0), t, equations)
        @test norm(SVector(u[1], u[2], u[3])) ≈ exp(-(t / width)^2)
        @test u[3] == 0
    end
    @test initial_condition_zero(x, t, heterogeneous) ==
          SVector(0, 0, 0, 0, 0, 0, 4, 1, 0)
end

@timed_testset "Hertzian dipole" begin
    equations = MaxwellEquations3D(epsilon = 2.0, mu = 1.5)
    c = speed_of_light(equations)
    Z = impedance(equations)
    signal = GaussianPulse(0.3; delay = 1.0)
    position = SVector(0.1, 0.0, -0.2)
    moment = SVector(0.3, 1.0, 0.5)
    field = HertzianDipoleField(position, moment, signal)
    E(x, t) = SVector(field(x, t, equations)[1:3]...)
    H(x, t) = SVector(field(x, t, equations)[4:6]...)

    # curl equations by finite differences away from the source
    h = 1.0e-4
    unit(i) = SVector{3}(ntuple(j -> j == i ? 1.0 : 0.0, 3))
    function curl(F, x, t)
        gradient = [(F(x + h * unit(j), t) - F(x - h * unit(j), t)) / (2h) for j in 1:3]
        return SVector(gradient[2][3] - gradient[3][2], gradient[3][1] - gradient[1][3],
                       gradient[1][2] - gradient[2][1])
    end
    for (x, t) in ((SVector(0.5, -0.4, 0.6), 1.9), (SVector(-0.3, 0.8, 0.1), 1.6))
        dEdt = (E(x, t + h) - E(x, t - h)) / (2h)
        dHdt = (H(x, t + h) - H(x, t - h)) / (2h)
        @test norm(curl(H, x, t) - equations.epsilon * dEdt) < 1e-5 * norm(dEdt)
        @test norm(curl(E, x, t) + equations.mu * dHdt) < 1e-5 * norm(dHdt)
        divergence = sum((E(x + h * unit(i), t)[i] - E(x - h * unit(i), t)[i]) / (2h)
                         for i in 1:3)
        @test abs(divergence) < 1e-4 * norm(E(x, t))
    end

    # far field: transverse, E = Z H x r, 1 / R decay along the equator of the dipole
    direction = SVector(1.0, 0.0, 0.0)
    axis = SVector(0.0, 0.0, 1.0)
    axial = HertzianDipoleField(SVector(0.0, 0.0, 0.0), axis, signal)
    for R in (20.0, 40.0)
        x = R * direction
        t = 1.0 + R / c
        u = axial(x, t, equations)
        E_far = SVector(u[1], u[2], u[3])
        H_far = SVector(u[4], u[5], u[6])
        @test abs(dot(E_far, direction)) < 1e-3 * norm(E_far)
        @test E_far≈Z * cross(H_far, direction) rtol=1e-2
        @test norm(E_far) *
              R≈norm(signal_second_derivative(signal, 1.0)) * equations.mu /
                (4 * pi) rtol=2e-2
    end

    # regularized dipole: the exterior field is the point dipole with a smoothed signal
    dipole = HertzianDipole(position, moment, 0.1, signal)
    smoothed = HertzianDipoleField(dipole, equations)
    smoothed_width = sqrt(0.3^2 + (0.1 / c)^2)
    @test smoothed.signal == GaussianPulse(smoothed_width; delay = 1.0)
    @test smoothed.moment ≈ moment * 0.3 / smoothed_width
    @test_throws ArgumentError HertzianDipoleField(HertzianDipole(position, moment, 0.1,
                                                                  ModulatedGaussianPulse(1.0,
                                                                                         0.3)),
                                                   equations)

    # source term: current density with unit integral and the Ohmic loss
    lossy = MaxwellEquations3D(epsilon = 2.0, mu = 1.5, sigma = 0.5)
    u = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    t = 0.7
    at_center = dipole(u, position, t, lossy)
    J0 = signal_derivative(signal, t) * moment / (sqrt(pi)^3 * 0.1^3)
    @test at_center[1:3] ≈ -J0 / 2.0 - 0.5 / 2.0 * u[1:3]
    @test at_center[4:6] == SVector(0.0, 0.0, 0.0)
    # Gauss-Hermite style check of the normalization on a fine grid
    grid = range(-0.5, 0.5, length = 201)
    dx = step(grid)
    total = sum(TrixiMaxwell.current_density(dipole, position + SVector(a, b, c), t)
                for a in grid, b in grid, c in grid) * dx^3
    @test total≈signal_derivative(signal, t) * moment rtol=1e-6
    heterogeneous = MaxwellEquations3D(Heterogeneous(); epsilon = 2.0, mu = 1.5)
    @test length(dipole(SVector(u..., 2.0, 1.5, 0.0), position, t, heterogeneous)) == 9
    @test dipole(SVector(u..., 2.0, 1.5, 0.0), position, t, heterogeneous)[7:9] ==
          SVector(0.0, 0.0, 0.0)
end

@timed_testset "PointEvaluator and TotalFieldScatteredField" begin
    solver = DGMulti(polydeg = 3, element_type = Tet(),
                     approximation_type = Polynomial(),
                     surface_integral = SurfaceIntegralWeakForm(flux_upwind),
                     volume_integral = VolumeIntegralWeakForm())
    mesh = DGMultiMesh(solver, (4, 4, 4); coordinates_min = (-1.0, -1.0, -1.0),
                       coordinates_max = (1.0, 1.0, 1.0),
                       periodicity = (false, false, false))
    polynomial(x, t, equations) = SVector(x[1], x[2], x[3], x[1] * x[2], x[3]^3, 1.0)
    semi = SemidiscretizationHyperbolic(mesh, MaxwellEquations3D(), polynomial, solver;
                                        boundary_conditions = (;
                                                               entire_boundary = boundary_condition_silver_mueller))
    ode = semidiscretize(semi, (0.0, 1.0))
    points = [
        SVector(0.3, -0.7, 0.123),
        SVector(-0.99, 0.5, 0.5),
        SVector(0.0, 0.0, 0.0)
    ]
    evaluator = PointEvaluator(points, semi)
    values = evaluator(ode.u0, semi)
    @test all(values[i] ≈ polynomial(points[i], 0.0, nothing)
              for i in eachindex(points))
    @test_throws ArgumentError PointEvaluator([SVector(1.5, 0.0, 0.0)], semi)

    wave = PlaneWave((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), GaussianPulse(0.2))
    tfsf = TotalFieldScatteredField(wave, mesh, x -> all(abs.(x) .< 0.5))
    # six box faces, 2 x 2 cells each, two triangles per cell, both sides
    @test length(tfsf.faces) == 6 * 4 * 2 * 2
    @test sum(tfsf.signs) == 0
    md = mesh.md
    for (face, sign) in zip(tfsf.faces, tfsf.signs)
        partner = findfirst(==(md.FToF[face]), tfsf.faces)
        @test partner !== nothing && tfsf.signs[partner] == -sign
    end
    @test_throws ArgumentError TotalFieldScatteredField(wave, mesh, x -> true)
    @test occursin("48 interface faces", repr(tfsf))
end

@timed_testset "PointEvaluator and ProjectedSourceTerms on DGSEM meshes" begin
    using LinearAlgebra: norm
    polydeg = 3
    solver = DGSEM(polydeg = polydeg, surface_flux = flux_upwind)
    polynomial(x, t, equations) = SVector(x[1], x[2], x[3], x[1] * x[2], x[3]^3, 1.0)
    points = [
        SVector(0.3, -0.7, 0.123),
        SVector(-0.99, 0.5, 0.5),
        SVector(0.0, 0.0, 0.0)
    ]
    pec = boundary_condition_perfect_electric_conductor
    meshes = (P4estMesh((2, 2, 2), polydeg = polydeg,
                        coordinates_min = (-1.0, -1.0, -1.0),
                        coordinates_max = (1.0, 1.0, 1.0), initial_refinement_level = 1,
                        periodicity = false),
              TreeMesh((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                       initial_refinement_level = 2,
                       periodicity = false))
    for mesh in meshes
        boundary_conditions = mesh isa TreeMesh ? pec :
                              (; x_neg = pec, x_pos = pec, y_neg = pec, y_pos = pec,
                               z_neg = pec, z_pos = pec)
        semi = SemidiscretizationHyperbolic(mesh, MaxwellEquations3D(), polynomial,
                                            solver;
                                            boundary_conditions)
        ode = semidiscretize(semi, (0.0, 1.0))
        values = PointEvaluator(points, semi)(ode.u0, semi)
        @test all(values[i] ≈ polynomial(points[i], 0.0, nothing)
                  for i in eachindex(points))
        @test_throws ArgumentError PointEvaluator([SVector(1.5, 0.0, 0.0)], semi)
    end

    # curved elements: the interpolant is no longer exact, the location still is
    warp(xi, eta, zeta) = SVector(xi, eta, zeta) .+
                          0.05 * sinpi(xi) * sinpi(eta) * sinpi(zeta)
    mesh = StructuredMesh((4, 4, 4), warp, periodicity = false)
    semi = SemidiscretizationHyperbolic(mesh, MaxwellEquations3D(), polynomial, solver;
                                        boundary_conditions = pec)
    ode = semidiscretize(semi, (0.0, 1.0))
    values = PointEvaluator(points, semi)(ode.u0, semi)
    @test all(isapprox(values[i], polynomial(points[i], 0.0, nothing); rtol = 1.0e-3)
              for i in eachindex(points))

    # sources of degree below polydeg are integrated exactly by the Lobatto rule,
    # so projection and collocation agree to rounding
    equations = MaxwellEquations3D()
    mesh = first(meshes)
    boundary_conditions = (; x_neg = pec, x_pos = pec, y_neg = pec, y_pos = pec,
                           z_neg = pec, z_pos = pec)
    smooth(u, x, t, equations) = SVector(x[1] * x[2], x[3]^2, 1.0, x[1], 0.0,
                                         x[2] + x[3])
    projected = ProjectedSourceTerms(smooth, equations, solver)
    @test occursin("quadrature points", repr(projected))
    semi_projected = SemidiscretizationHyperbolic(mesh, equations,
                                                  initial_condition_zero,
                                                  solver; boundary_conditions,
                                                  source_terms = projected)
    semi_pointwise = SemidiscretizationHyperbolic(mesh, equations,
                                                  initial_condition_zero,
                                                  solver; boundary_conditions,
                                                  source_terms = smooth)
    u0 = semidiscretize(semi_projected, (0.0, 1.0)).u0
    du_projected = similar(u0)
    du_pointwise = similar(u0)
    Trixi.rhs_hyperbolic!(du_projected, u0, semi_projected, 0.0)
    Trixi.rhs_hyperbolic!(du_pointwise, u0, semi_pointwise, 0.0)
    @test du_projected≈du_pointwise atol=1.0e-12
    @test projected(u0[1:6], points[1], 0.0, equations) ==
          smooth(u0[1:6], points[1], 0.0, equations)

    # a narrow dipole keeps its total current only when projected
    dipole = HertzianDipole((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.1,
                            GaussianPulse(0.4; delay = 1.4))
    t = 1.3
    expected = -SVector(0.0, 0.0, signal_derivative(dipole.signal, t)) # ∫ J dV / ε
    fine_mesh = P4estMesh((2, 2, 2), polydeg = polydeg,
                          coordinates_min = (-1.0, -1.0, -1.0),
                          coordinates_max = (1.0, 1.0, 1.0),
                          initial_refinement_level = 2,
                          periodicity = false)
    projected_dipole = ProjectedSourceTerms(dipole, equations, solver;
                                            quadrature_degree = 16)
    @test projected_dipole.radius == 0.5
    semi_fine = SemidiscretizationHyperbolic(fine_mesh, equations,
                                             initial_condition_zero,
                                             solver; boundary_conditions,
                                             source_terms = projected_dipole)
    u0_fine = semidiscretize(semi_fine, (0.0, 1.0)).u0
    du = similar(u0_fine)
    Trixi.rhs_hyperbolic!(du, u0_fine, semi_fine, t)
    current = Trixi.integrate((u, equations) -> u[1:3], du, semi_fine;
                              normalize = false)
    @test isapprox(current, expected; atol = 1.0e-4 * norm(expected))
    semi_fine = SemidiscretizationHyperbolic(fine_mesh, equations,
                                             initial_condition_zero,
                                             solver; boundary_conditions,
                                             source_terms = dipole)
    Trixi.rhs_hyperbolic!(du, u0_fine, semi_fine, t)
    current = Trixi.integrate((u, equations) -> u[1:3], du, semi_fine;
                              normalize = false)
    @test !isapprox(current, expected; atol = 1.0e-2 * norm(expected))
end

@timed_testset "Divergence diagnostics on DGSEM meshes" begin
    polydeg = 3
    solver = DGSEM(polydeg = polydeg, surface_flux = flux_upwind)
    equations = MaxwellEquations3D()
    # div E = 3, div H = 2x
    fields(x, t, equations) = SVector(x[1], x[2], x[3], x[1]^2, 0.0, 0.0)
    pec = boundary_condition_perfect_electric_conductor
    sides = (; x_neg = pec, x_pos = pec, y_neg = pec, y_pos = pec, z_neg = pec,
             z_pos = pec)
    warp(xi, eta, zeta) = SVector(xi, eta, zeta) .+
                          0.05 * sinpi(xi) * sinpi(eta) * sinpi(zeta)
    cases = ((P4estMesh((2, 2, 2), polydeg = polydeg,
                        coordinates_min = (-1.0, -1.0, -1.0),
                        coordinates_max = (1.0, 1.0, 1.0), initial_refinement_level = 1,
                        periodicity = false), sides),
             (TreeMesh((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                       initial_refinement_level = 2,
                       periodicity = false), pec),
             (StructuredMesh((4, 4, 4), warp, periodicity = false), pec))
    for (mesh, boundary_conditions) in cases
        semi = SemidiscretizationHyperbolic(mesh, equations, fields, solver;
                                            boundary_conditions)
        u0 = semidiscretize(semi, (0.0, 1.0)).u0
        u = Trixi.wrap_array(u0, semi)
        _, _, _, cache = Trixi.mesh_equations_solver_cache(semi)
        function analyze(name)
            Trixi.analyze(Val(name), nothing, u, 0.0, mesh, equations,
                          solver, cache)
        end
        # the interpolated metric terms of the warped mesh alias the products
        rtol = mesh isa StructuredMesh ? 5.0e-2 : 1.0e-12
        @test isapprox(analyze(:l2_dive), 3 * sqrt(8.0); rtol)
        @test isapprox(analyze(:linf_dive), 3.0; rtol)
        @test isapprox(analyze(:l2_divh), sqrt(4 * 2 / 3 * 4); rtol)
        @test isapprox(analyze(:linf_divh), 2.0; rtol)
    end
end

@timed_testset "Heterogeneous interface fluxes on DGSEM meshes" begin
    using LinearAlgebra: norm
    polydeg = 3
    solver = DGSEM(polydeg = polydeg, surface_flux = flux_upwind)
    equations = MaxwellEquations3D(Heterogeneous())
    # smooth fields with a material jump at x = 0 on element faces
    fields(x, t, equations) = SVector(sinpi(x[2]) * cospi(x[3]), cospi(x[1]),
                                      sinpi(x[1] + x[3]),
                                      cospi(x[2]), sinpi(x[3]), cospi(x[1] - x[2]),
                                      1.0, 1.0, 0.0)
    material_at(x) = x[1] < 0 ? Material(epsilon = 1.0) : Material(epsilon = 4.0)
    pec = boundary_condition_perfect_electric_conductor
    sides = (; x_neg = pec, x_pos = pec, y_neg = pec, y_pos = pec, z_neg = pec,
             z_pos = pec)
    lo, hi = (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)
    meshes = ((TreeMesh(lo, hi, initial_refinement_level = 2, periodicity = false),
               pec),
              (P4estMesh((4, 4, 4), polydeg = polydeg, coordinates_min = lo,
                         coordinates_max = hi, initial_refinement_level = 0,
                         periodicity = false), sides),
              (T8codeMesh((4, 4, 4), polydeg = polydeg, coordinates_min = lo,
                          coordinates_max = hi, initial_refinement_level = 0,
                          periodicity = false), sides))
    points = [SVector(0.1, -0.3, 0.2), SVector(-0.05, 0.4, -0.6),
        SVector(0.49, 0.01, 0.0),
        SVector(-0.9, -0.9, 0.9)]
    results = map(meshes) do (mesh, boundary_conditions)
        semi = SemidiscretizationHyperbolic(mesh, equations, fields, solver;
                                            boundary_conditions)
        ode = semidiscretize(semi, (0.0, 1.0))
        set_materials!(ode.u0, semi, material_at)
        du = similar(ode.u0)
        Trixi.rhs_hyperbolic!(du, ode.u0, semi, 0.0)
        PointEvaluator(points, semi)(du, semi)
    end
    for other in results[2:end]
        @test all(isapprox(a, b; atol = 1.0e-10 * norm(a))
                  for (a, b) in zip(results[1], other))
    end
    # the material jump changes the flux: without it the right-hand side differs
    mesh, boundary_conditions = meshes[2]
    semi = SemidiscretizationHyperbolic(mesh, equations, fields, solver;
                                        boundary_conditions)
    ode = semidiscretize(semi, (0.0, 1.0))
    du = similar(ode.u0)
    Trixi.rhs_hyperbolic!(du, ode.u0, semi, 0.0)
    homogeneous = PointEvaluator(points, semi)(du, semi)
    @test !all(isapprox(a, b; atol = 1.0e-6 * norm(a))
               for (a, b) in zip(results[2], homogeneous))
end

@timed_testset "Uniaxial PML" begin
    equations = MaxwellEquations3D(UPML(); epsilon = 2.0, mu = 1.5)
    @test equations isa MaxwellEquations3D{Homogeneous, UPML, 12, Float64}
    @test MaxwellEquations3D(Heterogeneous(), UPML()) isa
          MaxwellEquations3D{Heterogeneous, UPML, 15, Float64}
    @test_throws MethodError MaxwellEquations3D(UPML(), Heterogeneous())
    @test Trixi.varnames(cons2cons, equations) ==
          ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "px", "py", "pz", "qx", "qy", "qz")
    @test Trixi.varnames(cons2cons, MaxwellEquations3D(Heterogeneous(), UPML()))[7:9] ==
          ("epsilon", "mu", "sigma")
    @test TrixiMaxwell.pml_offset(equations) == 6
    @test TrixiMaxwell.pml_offset(MaxwellEquations3D(Heterogeneous(), UPML())) == 9

    fields = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    u = TrixiMaxwell.with_passive_defaults(fields, equations)
    @test u == vcat(fields, zeros(6))
    @test TrixiMaxwell.passive_flux(equations) == zeros(6)
    @test TrixiMaxwell.pml_electric(u, equations) == zeros(3)
    u = SVector(ntuple(Float64, 12))
    @test TrixiMaxwell.pml_electric(u, equations) == SVector(7.0, 8.0, 9.0)
    @test TrixiMaxwell.pml_magnetic(u, equations) == SVector(10.0, 11.0, 12.0)
    @test TrixiMaxwell.assemble(SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), u,
                                equations) ==
          vcat(zeros(6), u[7:12])
    @test flux(u, 1, equations)[7:12] == zeros(6)
    @test flux_upwind(u, u, SVector(0.0, 1.0, 0.0), equations)[7:12] == zeros(6)
    @test cons2entropy(u, equations)[7:12] == zeros(6)
    @test energy_total(u, equations) ==
          energy_total(fields, MaxwellEquations3D(epsilon = 2.0, mu = 1.5))

    # profile: zero inside, polynomial depth in each layer, corners add up per axis
    profile = PMLProfile((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 0.5)
    @test profile.sigma_max ≈ 4 * log(1e6) / (2 * 0.5)
    @test profile(SVector(0.9, -0.9, 0.0)) == zeros(3)
    @test profile(SVector(1.25, 0.0, 0.0)) ≈ SVector(profile.sigma_max / 8, 0.0, 0.0)
    @test profile(SVector(-1.5, 1.5, 1.0)) ≈
          SVector(profile.sigma_max, profile.sigma_max, 0.0)
    @test PMLProfile((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.25;
                     sigma_max = 3.0).sigma_max == 3.0
    @test PMLProfile((0.0f0, 0.0f0, 0.0f0), (1.0f0, 1.0f0, 1.0f0), 0.25f0) isa
          PMLProfile{Float32}

    # source: eq. 38 at a node with sigma = (s, 0, 0)
    source = SourceTermsPML(profile)
    x = SVector(1.25, 0.0, 0.0)
    s = profile(x)[1]
    E, H = SVector(u[1:3]...), SVector(u[4:6]...)
    p, q = SVector(u[7:9]...), SVector(u[10:12]...)
    du = source(u, x, 0.0, equations)
    damping = SVector(-s, s, s)
    coupling = SVector(s^2, 0.0, 0.0)
    @test du[1:3] ≈ -damping .* E - p / 2.0
    @test du[4:6] ≈ -damping .* H - q / 1.5
    @test du[7:9] ≈ coupling .* (2.0 * E) - SVector(s, 0.0, 0.0) .* p
    @test du[10:12] ≈ coupling .* (1.5 * H) - SVector(s, 0.0, 0.0) .* q
    @test all(iszero,
              source(vcat(fields, zeros(SVector{6})), SVector(0.0, 0.0, 0.0), 0.0,
                     equations))
    @test_throws ArgumentError source(fields, x, 0.0, MaxwellEquations3D())
    heterogeneous = MaxwellEquations3D(Heterogeneous(), UPML())
    u_het = vcat(fields, SVector(2.0, 1.5, 0.0), u[7:12])
    @test source(u_het, x, 0.0, heterogeneous)[7:9] == zeros(3)
    @test source(u_het, x, 0.0, heterogeneous)[10:15] ≈ du[7:12]

    combined = CombinedSourceTerms(source, source_terms_conductivity)
    lossy = MaxwellEquations3D(UPML(); epsilon = 2.0, mu = 1.5, sigma = 0.5)
    @test combined(u, x, 0.0, lossy) ≈
          source(u, x, 0.0, lossy) + source_terms_conductivity(u, x, 0.0, lossy)
end
end

end # module
