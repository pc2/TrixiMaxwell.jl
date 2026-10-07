"""
    TotalFieldScatteredField(incident_field, mesh::DGMultiMesh, is_total_field)

Total-field/scattered-field interface on all interior faces between elements
whose centroids `is_total_field(x)` classifies differently. Used as a value of
`boundary_conditions`, it recomputes the interface flux on these faces with the
incident field added to the neighbor state seen from the total-field side and
subtracted on the scattered-field side, so the incident wave exists only in the
total-field region. The incident field must be a free-space solution in the
background material, which has to fill the interface faces. Start from
[`initial_condition_zero`](@ref) with a delayed signal, or with the incident
field if it is still outside the domain.
"""
struct TotalFieldScatteredField{IncidentField}
    incident_field::IncidentField
    faces::Vector{Int}
    signs::Vector{Int}
end

function TotalFieldScatteredField(incident_field, mesh::DGMultiMesh{3}, is_total_field)
    md = mesh.md
    num_faces = size(md.FToF, 1)
    total_field = [is_total_field(element_centroid(md, element))
                   for element in Base.OneTo(md.num_elements)]

    faces = Int[]
    signs = Int[]
    for face in eachindex(md.FToF)
        neighbor = md.FToF[face]
        is_boundary_face(md, face) && continue
        element = (face - 1) ÷ num_faces + 1
        neighbor_element = (neighbor - 1) ÷ num_faces + 1
        total_field[element] == total_field[neighbor_element] && continue
        push!(faces, face)
        push!(signs, total_field[element] ? 1 : -1)
    end
    isempty(faces) &&
        throw(ArgumentError("is_total_field does not separate the mesh into two regions"))

    return TotalFieldScatteredField(incident_field, faces, signs)
end

function Base.show(io::IO, tfsf::TotalFieldScatteredField)
    print(io, "TotalFieldScatteredField(", tfsf.incident_field, ", ",
          length(tfsf.faces) ÷ 2, " interface faces)")
end

# The interface flux has been computed for all faces at this point; the
# boundary flux stage is the first one that sees `t` and the face coordinates.
function Trixi.calc_single_boundary_flux!(cache, t, tfsf::TotalFieldScatteredField,
                                          boundary_key, mesh::DGMultiMesh,
                                          have_nonconservative_terms::Trixi.False,
                                          equations::MaxwellEquations3D, dg::DGMulti{3})
    rd = dg.basis
    md = mesh.md
    (; u_face_values, flux_face_values) = cache.solution_container
    (; mapP, xyzf, nxyzJ, Jf) = md
    (; surface_flux) = dg.surface_integral
    num_pts_per_face = rd.Nfq ÷ StartUpDG.num_faces(rd.element_type)
    (; faces, signs, incident_field) = tfsf

    Trixi.@threaded for index in eachindex(faces)
        face = faces[index]
        sign = signs[index]
        for i in Base.OneTo(num_pts_per_face)
            idM = (face - 1) * num_pts_per_face + i
            idP = mapP[idM]
            x = SVector{3}(getindex.(xyzf, idM))
            normal = SVector{3}(getindex.(nxyzJ, idM)) / Jf[idM]

            u_incident = incident_field(x, t, equations)
            uM = u_face_values[idM]
            uP = u_face_values[idP]
            uP_corrected = assemble(electric_field(uP) + sign * electric_field(u_incident),
                                    magnetic_field(uP) + sign * magnetic_field(u_incident),
                                    uP, equations)
            flux_face_values[idM] = surface_flux(uM, uP_corrected, normal, equations) *
                                    Jf[idM]
        end
    end

    return nothing
end
