@doc raw"""
    SurfaceTransformCallback

Discrete Fourier transform ``\hat u(f) = \sum_n u(t_n) e^{-2 \pi i f t_n} (t_n - t_{n-1})``
of the scattered field ``u - u_i`` and of the incident field ``u_i`` at the face
nodes of an oriented surface of interior faces, accumulated after every time
step. Built by [`CrossSectionCallback`](@ref) and [`DetectorPlaneCallback`](@ref);
evaluated by [`cross_sections`](@ref) and [`transmittance_reflectance`](@ref).
The fields have to have left the surface at the end of the simulation.
"""
mutable struct SurfaceTransformCallback{RealT <: Real, IncidentField}
    frequencies::Vector{RealT}
    incident_field::IncidentField
    face_nodes::Vector{NTuple{4, Int}} # (node, element) on the inner and the outer side
    incident_on_sides::NTuple{2, RealT} # 1 if the side carries the total field, 0 if the scattered field
    coordinates::Vector{SVector{3, RealT}}
    weighted_normals::Vector{SVector{3, RealT}} # outward normal times quadrature weight
    interpolation::Matrix{RealT}
    admittance::RealT
    scattered::Array{Complex{RealT}, 3}
    incident::Array{Complex{RealT}, 3}
    t_previous::RealT
end

# `faces` are face ids of the inner elements; their normals point outwards
function SurfaceTransformCallback(semi, incident_field, frequencies, faces,
                                  incident_on_sides)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    rd = dg.basis
    md = mesh.md
    RealT = eltype(md.xyzf[1])
    (; mapP, xyzf, nxyzJ) = md
    num_pts_per_face = rd.Nfq ÷ StartUpDG.num_faces(rd.element_type)
    face_weights = reshape(rd.wf, :)

    face_nodes = NTuple{4, Int}[]
    coordinates = SVector{3, RealT}[]
    weighted_normals = SVector{3, RealT}[]
    for face in faces, i in Base.OneTo(num_pts_per_face)
        idM = (face - 1) * num_pts_per_face + i
        idP = mapP[idM]
        push!(face_nodes,
              ((idM - 1) % rd.Nfq + 1, (idM - 1) ÷ rd.Nfq + 1,
               (idP - 1) % rd.Nfq + 1, (idP - 1) ÷ rd.Nfq + 1))
        push!(coordinates, SVector{3}(getindex.(xyzf, idM)))
        push!(weighted_normals,
              face_weights[(idM - 1) % rd.Nfq + 1] * SVector{3}(getindex.(nxyzJ, idM)))
    end

    frequencies = collect(RealT, frequencies)
    num_nodes = length(face_nodes)
    transform = SurfaceTransformCallback(frequencies, incident_field, face_nodes,
                                         convert.(RealT, incident_on_sides), coordinates,
                                         weighted_normals, Matrix{RealT}(rd.Vf),
                                         convert(RealT, equations.admittance),
                                         zeros(Complex{RealT}, 6, num_nodes,
                                               length(frequencies)),
                                         zeros(Complex{RealT}, 6, num_nodes,
                                               length(frequencies)),
                                         zero(RealT))
    return Trixi.DiscreteCallback(transform, transform, save_positions = (false, false),
                                  initialize = initialize_surface_transform!)
end

function check_dgmulti(semi, name)
    mesh, _, _, _ = Trixi.mesh_equations_solver_cache(semi)
    mesh isa DGMultiMesh ||
        throw(ArgumentError("$name needs a DGMultiMesh, got $(typeof(mesh))"))
    return mesh
end

"""
    CrossSectionCallback(semi, tfsf::TotalFieldScatteredField, frequencies)

[`SurfaceTransformCallback`](@ref) on the surface of the total-field region of
`tfsf`, for [`cross_sections`](@ref). The scattered field on the surface is the
mean of the scattered-field trace and the total-field trace minus the incident
field.
"""
function CrossSectionCallback(semi, tfsf::TotalFieldScatteredField, frequencies)
    check_dgmulti(semi, "CrossSectionCallback")
    faces = [face for (face, sign) in zip(tfsf.faces, tfsf.signs) if sign > 0]
    return SurfaceTransformCallback(semi, tfsf.incident_field, frequencies, faces, (1, 0))
end

"""
    DetectorPlaneCallback(semi, incident_field, frequencies; point, normal)

[`SurfaceTransformCallback`](@ref) on all interior faces in the plane through
`point` with unit `normal`, oriented along `normal`, for
[`transmittance_reflectance`](@ref). The plane has to lie in the total-field
region and consist of element faces.
"""
function DetectorPlaneCallback(semi, incident_field, frequencies; point, normal)
    mesh = check_dgmulti(semi, "DetectorPlaneCallback")
    _, _, dg, _ = Trixi.mesh_equations_solver_cache(semi)
    rd = dg.basis
    md = mesh.md
    (; xyzf, nxyzJ, Jf) = md
    num_pts_per_face = rd.Nfq ÷ StartUpDG.num_faces(rd.element_type)
    point = SVector{3}(point)
    normal = SVector{3}(normal) / norm(normal)
    scale = maximum(maximum(abs, x) for x in md.xyz)
    tolerance = 100 * eps(eltype(normal)) * scale

    faces = Int[]
    for face in eachindex(md.FToF)
        is_boundary_face(md, face) && continue
        ids = (face - 1) * num_pts_per_face .+ (1:num_pts_per_face)
        all(abs(dot(SVector{3}(getindex.(xyzf, id)) - point, normal)) < tolerance
            for id in ids) || continue
        id = first(ids)
        dot(SVector{3}(getindex.(nxyzJ, id)), normal) > Jf[id] / 2 && push!(faces, face)
    end
    isempty(faces) &&
        throw(ArgumentError("no interior element faces lie in the plane through $point with normal $normal"))
    return SurfaceTransformCallback(semi, incident_field, frequencies, faces, (1, 1))
end

function initialize_surface_transform!(cb, u, t, integrator)
    transform = cb.affect!
    fill!(transform.scattered, 0)
    fill!(transform.incident, 0)
    transform.t_previous = t
    return nothing
end

# condition
(transform::SurfaceTransformCallback)(u, t, integrator) = true

# affect!
function (transform::SurfaceTransformCallback)(integrator)
    semi = integrator.p
    _, equations, _, _ = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(integrator.u, semi)
    t = integrator.t
    dt = t - transform.t_previous
    transform.t_previous = t
    (; frequencies, incident_field, face_nodes, coordinates, interpolation,
    scattered, incident) = transform
    incident_inner, incident_outer = transform.incident_on_sides

    Trixi.@threaded for node in eachindex(face_nodes)
        i_inner, element_inner, i_outer, element_outer = face_nodes[node]
        u_inner = face_value(u, interpolation, i_inner, element_inner)
        u_outer = face_value(u, interpolation, i_outer, element_outer)
        u_incident = incident_field(coordinates[node], t, equations)
        fields_incident = vcat(electric_field(u_incident), magnetic_field(u_incident))
        fields_scattered = (vcat(electric_field(u_inner), magnetic_field(u_inner)) +
                            vcat(electric_field(u_outer), magnetic_field(u_outer)) -
                            (incident_inner + incident_outer) * fields_incident) / 2
        for k in eachindex(frequencies)
            phase = cis(-2 * oftype(t, pi) * frequencies[k] * t) * dt
            for v in 1:6
                scattered[v, node, k] += fields_scattered[v] * phase
                incident[v, node, k] += fields_incident[v] * phase
            end
        end
    end

    Trixi.derivative_discontinuity!(integrator, false)
    return nothing
end

@inline function face_value(u, interpolation, i, element)
    value = interpolation[i, 1] * u[1, element]
    for n in 2:size(interpolation, 2)
        value += interpolation[i, n] * u[n, element]
    end
    return value
end

# Time-averaged Poynting fluxes through the surface, up to the common factor 1/2:
# of the scattered field, of the cross terms between scattered and incident
# field, and of the incident field, plus the mean incident intensity.
function surface_fluxes(transform::SurfaceTransformCallback, k)
    (; weighted_normals, admittance, scattered, incident) = transform
    RealT = eltype(transform.frequencies)
    flux_scattered = zero(RealT)
    flux_cross = zero(RealT)
    flux_incident = zero(RealT)
    intensity = zero(RealT)
    area = zero(RealT)
    for node in eachindex(weighted_normals)
        normal = weighted_normals[node]
        E_s = SVector{3}(view(scattered, 1:3, node, k))
        H_s = SVector{3}(view(scattered, 4:6, node, k))
        E_i = SVector{3}(view(incident, 1:3, node, k))
        H_i = SVector{3}(view(incident, 4:6, node, k))
        flux_scattered += dot(real(cross(E_s, conj(H_s))), normal)
        flux_cross += dot(real(cross(E_s, conj(H_i)) + cross(E_i, conj(H_s))), normal)
        flux_incident += dot(real(cross(E_i, conj(H_i))), normal)
        intensity += admittance * sum(abs2, E_i) * norm(normal)
        area += norm(normal)
    end
    return (; flux_scattered, flux_cross, flux_incident, intensity = intensity / area)
end

@doc raw"""
    cross_sections(callback)

Scattering, extinction and absorption cross sections at the frequencies of a
[`CrossSectionCallback`](@ref), from the transformed fields on the closed surface:
```math
\sigma_\mathrm{sca} = \frac{1}{Y |\hat E_i|^2} \oint \operatorname{Re}(\hat E_s \times \hat H_s^*) \cdot n \, dA,
\quad
\sigma_\mathrm{ext} = -\frac{1}{Y |\hat E_i|^2} \oint \operatorname{Re}(\hat E_s \times \hat H_i^* + \hat E_i \times \hat H_s^*) \cdot n \, dA,
```
with the incident intensity averaged over the surface and
``\sigma_\mathrm{abs} = \sigma_\mathrm{ext} - \sigma_\mathrm{sca}``.
Returns a named tuple `(; frequencies, scattering, extinction, absorption)`.
"""
function cross_sections(cb::Trixi.DiscreteCallback{<:Any, <:SurfaceTransformCallback})
    (; frequencies) = cb.affect!
    fluxes = [surface_fluxes(cb.affect!, k) for k in eachindex(frequencies)]
    scattering = [flux.flux_scattered / flux.intensity for flux in fluxes]
    extinction = [-flux.flux_cross / flux.intensity for flux in fluxes]
    return (; frequencies, scattering, extinction, absorption = extinction - scattering)
end

@doc raw"""
    transmittance_reflectance(callback)

Power transmittance and reflectance at the frequencies of a
[`DetectorPlaneCallback`](@ref): the flux of the total field and the negative
flux of the scattered field through the plane, both divided by the flux of the
incident field,
```math
T = \frac{\int \operatorname{Re}(\hat E \times \hat H^*) \cdot n \, dA}{\int \operatorname{Re}(\hat E_i \times \hat H_i^*) \cdot n \, dA},
\quad
R = -\frac{\int \operatorname{Re}(\hat E_s \times \hat H_s^*) \cdot n \, dA}{\int \operatorname{Re}(\hat E_i \times \hat H_i^*) \cdot n \, dA}.
```
``T`` is meaningful behind a scatterer, ``R`` in front of it.
Returns a named tuple `(; frequencies, transmittance, reflectance)`.
"""
function transmittance_reflectance(cb::Trixi.DiscreteCallback{<:Any,
                                                              <:SurfaceTransformCallback})
    (; frequencies) = cb.affect!
    fluxes = [surface_fluxes(cb.affect!, k) for k in eachindex(frequencies)]
    transmittance = [(flux.flux_scattered + flux.flux_cross + flux.flux_incident) /
                     flux.flux_incident for flux in fluxes]
    reflectance = [-flux.flux_scattered / flux.flux_incident for flux in fluxes]
    return (; frequencies, transmittance, reflectance)
end

function Base.show(io::IO, cb::Trixi.DiscreteCallback{<:Any, <:SurfaceTransformCallback})
    @nospecialize cb
    transform = cb.affect!
    print(io, "SurfaceTransformCallback(", length(transform.frequencies),
          " frequencies, ", length(transform.face_nodes), " surface nodes)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::Trixi.DiscreteCallback{<:Any, <:SurfaceTransformCallback})
    @nospecialize cb
    if get(io, :compact, false)
        show(io, cb)
    else
        transform = cb.affect!
        setup = [
            "frequencies" => "$(length(transform.frequencies)) in [$(first(transform.frequencies)), $(last(transform.frequencies))]",
            "surface nodes" => length(transform.face_nodes),
            "incident field" => transform.incident_field]
        Trixi.summary_box(io, "SurfaceTransformCallback", setup)
    end
    return nothing
end
