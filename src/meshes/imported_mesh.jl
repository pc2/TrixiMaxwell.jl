"""
    ImportedMesh(vertex_coordinates, EToV; element_groups, group_names,
                 face_sets, face_set_names, edge_nodes)

Linear tetrahedral mesh as read from a file, before any solver-specific
processing. `vertex_coordinates` is a tuple of three coordinate vectors, `EToV`
the `K x 4` element-to-vertex connectivity. `element_groups` holds one integer
tag per element and `face_sets` maps an integer tag to the faces of that set,
each face given by its three vertex ids. Names are optional and map a tag to a
string. `edge_nodes` maps a sorted pair of vertex ids to the coordinates of the
mid-edge node of a quadratic tetrahedron; it is empty for straight-sided meshes.
Tetrahedra with negative Jacobian are reoriented on construction.

Build a solver mesh with [`DGMultiMesh`](@ref)`(dg, imported)`, curved if
`edge_nodes` is not empty.
"""
struct ImportedMesh{RealT <: Real}
    vertex_coordinates::NTuple{3, Vector{RealT}}
    EToV::Matrix{Int}
    element_groups::Vector{Int}
    group_names::Dict{Int, String}
    face_sets::Dict{Int, Vector{NTuple{3, Int}}}
    face_set_names::Dict{Int, String}
    edge_nodes::Dict{NTuple{2, Int}, NTuple{3, RealT}}
end

function ImportedMesh(vertex_coordinates::NTuple{3, AbstractVector}, EToV::AbstractMatrix;
                      element_groups = ones(Int, size(EToV, 1)),
                      group_names = Dict{Int, String}(),
                      face_sets = Dict{Int, Vector{NTuple{3, Int}}}(),
                      face_set_names = Dict{Int, String}(),
                      edge_nodes = Dict{NTuple{2, Int}, NTuple{3, Float64}}())
    VX, VY, VZ = vertex_coordinates
    num_vertices = length(VX)
    if length(VY) != num_vertices || length(VZ) != num_vertices
        throw(ArgumentError("coordinate vectors have different lengths: $num_vertices, $(length(VY)), $(length(VZ))"))
    end
    if size(EToV, 2) != 4
        throw(ArgumentError("EToV must have four columns for tetrahedra, got $(size(EToV, 2))"))
    end
    num_elements = size(EToV, 1)
    if !all(v -> 1 <= v <= num_vertices, EToV)
        throw(ArgumentError("EToV contains vertex ids outside 1:$num_vertices"))
    end
    if length(element_groups) != num_elements
        throw(ArgumentError("element_groups has length $(length(element_groups)), expected $num_elements"))
    end
    for tag in keys(group_names)
        tag in element_groups ||
            throw(ArgumentError("group name given for tag $tag, but no element carries it"))
    end
    for tag in keys(face_set_names)
        haskey(face_sets, tag) ||
            throw(ArgumentError("face set name given for tag $tag, but no face set carries it"))
    end

    RealT = promote_type(eltype(VX), eltype(VY), eltype(VZ))
    coordinates = (Vector{RealT}(VX), Vector{RealT}(VY), Vector{RealT}(VZ))
    connectivity = Matrix{Int}(EToV)
    StartUpDG.correct_negative_tet_jacobians!(coordinates, connectivity)

    sorted_sets = Dict{Int, Vector{NTuple{3, Int}}}()
    for (tag, faces) in face_sets
        sorted_sets[tag] = [Tuple(sort(collect(face))) for face in faces]
    end

    sorted_edges = Dict{NTuple{2, Int}, NTuple{3, RealT}}()
    for (edge, node) in edge_nodes
        all(v -> 1 <= v <= num_vertices, edge) ||
            throw(ArgumentError("edge node given for edge $edge with vertex ids outside 1:$num_vertices"))
        sorted_edges[minmax(edge...)] = NTuple{3, RealT}(node)
    end
    if !isempty(sorted_edges)
        for element in axes(connectivity, 1), i in 1:4, j in (i + 1):4
            edge = minmax(connectivity[element, i], connectivity[element, j])
            haskey(sorted_edges, edge) ||
                throw(ArgumentError("edge $edge of element $element has no edge node"))
        end
    end

    return ImportedMesh{RealT}(coordinates, connectivity, Vector{Int}(element_groups),
                               Dict{Int, String}(group_names), sorted_sets,
                               Dict{Int, String}(face_set_names), sorted_edges)
end

# Plain mesh data as returned by the file readers: a NamedTuple with the fields of
# `ImportedMesh`, built from base Julia arrays and dictionaries only.
function ImportedMesh(data::NamedTuple)
    return ImportedMesh(data.vertex_coordinates, data.EToV;
                        element_groups = data.element_groups,
                        group_names = data.group_names,
                        face_sets = data.face_sets,
                        face_set_names = data.face_set_names,
                        edge_nodes = get(data, :edge_nodes,
                                         Dict{NTuple{2, Int}, NTuple{3, Float64}}()))
end

num_vertices(imported::ImportedMesh) = length(first(imported.vertex_coordinates))
num_elements(imported::ImportedMesh) = size(imported.EToV, 1)

function Base.show(io::IO, imported::ImportedMesh{RealT}) where {RealT}
    print(io, "ImportedMesh{", RealT, "} with ", num_vertices(imported), " vertices, ",
          num_elements(imported), isempty(imported.edge_nodes) ? "" : " quadratic",
          " tetrahedra, ",
          length(unique(imported.element_groups)), " element groups, ",
          length(imported.face_sets), " face sets")
end

"""
    face_set_key(tag, names)

Key used for a face set in `boundary_conditions`: the set name as a `Symbol`
if one exists, otherwise `tag_<tag>`.
"""
function face_set_key(tag::Integer, names::Dict{Int, String})
    return haskey(names, tag) ? Symbol(names[tag]) : Symbol("tag_", tag)
end

# Global face id used by StartUpDG and Trixi: faces of an element are numbered
# consecutively in the order of `rd.fv`.
global_face_id(element, face, num_faces) = face + (element - 1) * num_faces

function face_lookup(EToV, fv)
    lookup = Dict{NTuple{3, Int}, Int}()
    num_faces = length(fv)
    for element in axes(EToV, 1), (face, vertices) in enumerate(fv)
        triple = Tuple(sort([EToV[element, v] for v in vertices]))
        lookup[triple] = global_face_id(element, face, num_faces)
    end
    return lookup
end

is_boundary_face(md, face_id) = md.FToF[face_id] == face_id

"""
    boundary_face_sets(imported, md, fv; extra = NamedTuple(), allow_untagged_boundary = false)

Map the face sets of `imported` to global face ids of the `MeshData` `md`,
using the reference element face vertices `fv`. Sets consisting of interior
faces only are skipped; a set mixing boundary and interior faces is an error.
`extra` adds face sets that were tagged by coordinate predicates. Every
boundary face must end up in exactly one set unless `allow_untagged_boundary`
is set.
"""
function boundary_face_sets(imported::ImportedMesh, md, fv;
                            extra = NamedTuple(), allow_untagged_boundary = false)
    lookup = face_lookup(imported.EToV, fv)
    owner = Dict{Int, Symbol}()
    sets = Dict{Symbol, Vector{Int}}()

    function assign!(key, face_ids)
        haskey(sets, key) &&
            throw(ArgumentError("face set $key is defined twice"))
        for face_id in face_ids
            if haskey(owner, face_id)
                throw(ArgumentError("boundary face $face_id belongs to the sets $(owner[face_id]) and $key"))
            end
            owner[face_id] = key
        end
        sets[key] = sort(collect(face_ids))
        return nothing
    end

    for tag in sort(collect(keys(imported.face_sets)))
        key = face_set_key(tag, imported.face_set_names)
        boundary_ids = Int[]
        num_interior = 0
        for triple in imported.face_sets[tag]
            face_id = get(lookup, triple, 0)
            face_id == 0 &&
                throw(ArgumentError("face $triple of set $key is not a face of any element"))
            if is_boundary_face(md, face_id)
                push!(boundary_ids, face_id)
            else
                num_interior += 1
            end
        end
        if num_interior > 0 && !isempty(boundary_ids)
            throw(ArgumentError("face set $key contains $(length(boundary_ids)) boundary faces and $num_interior interior faces"))
        end
        isempty(boundary_ids) || assign!(key, boundary_ids)
    end

    for (key, face_ids) in pairs(extra)
        assign!(key, face_ids)
    end

    num_untagged = count(face_id -> is_boundary_face(md, face_id) && !haskey(owner, face_id),
                         1:length(md.FToF))
    if num_untagged > 0 && !allow_untagged_boundary
        throw(ArgumentError("$num_untagged boundary faces belong to no face set; tag them in the mesh file, pass `is_on_boundary`, or set `allow_untagged_boundary = true`"))
    end

    keys_sorted = sort(collect(keys(sets)))
    return NamedTuple{Tuple(keys_sorted)}(Tuple(sets[key] for key in keys_sorted))
end

"""
    DGMultiMesh(dg::DGMulti, imported::ImportedMesh;
                is_on_boundary = nothing, allow_untagged_boundary = false)

Build the solver mesh of an imported tetrahedral mesh. Face sets of the file
become boundary keys, see [`face_set_key`](@ref). `is_on_boundary` adds
coordinate-predicate boundaries in the style of Trixi's `DGMultiMesh`. With
edge nodes, the elements are mapped by their quadratic shape functions and the
mesh is `Curved()`.
"""
function Trixi.DGMultiMesh(dg::DGMulti{3}, imported::ImportedMesh;
                           is_on_boundary = nothing, allow_untagged_boundary = false)
    rd = dg.basis
    if !(rd.element_type isa StartUpDG.Tet)
        throw(ArgumentError("imported meshes consist of tetrahedra, but the solver uses $(rd.element_type)"))
    end
    VX, VY, VZ = imported.vertex_coordinates
    md = StartUpDG.MeshData(VX, VY, VZ, imported.EToV, rd)
    extra = is_on_boundary === nothing ? NamedTuple() :
            StartUpDG.tag_boundary_faces(md, is_on_boundary)
    boundary_faces = boundary_face_sets(imported, md, rd.fv; extra, allow_untagged_boundary)
    if isempty(imported.edge_nodes)
        return Trixi.DGMultiMesh(dg, Trixi.GeometricTermsType(Trixi.VertexMapped(), dg),
                                 md, boundary_faces)
    end
    md_curved = StartUpDG.MeshData(rd, md, quadratic_node_coordinates(imported, rd)...)
    minimum(md_curved.J) > 0 ||
        throw(ArgumentError("the quadratic elements have a non-positive Jacobian"))
    return Trixi.DGMultiMesh(dg, Trixi.GeometricTermsType(Trixi.Curved(), dg), md_curved,
                             boundary_faces)
end

# Coordinates of the nodes of `rd` under the quadratic map of every element, with
# the vertices of `EToV` at the reference vertices (-1, -1, -1), (1, -1, -1),
# (-1, 1, -1) and (-1, -1, 1).
function quadratic_node_coordinates(imported::ImportedMesh, rd)
    (; EToV, edge_nodes) = imported
    VX, VY, VZ = imported.vertex_coordinates
    r, s, t = rd.rst
    barycentric = (-(1 .+ r .+ s .+ t) / 2, (1 .+ r) / 2, (1 .+ s) / 2, (1 .+ t) / 2)
    x, y, z = (zeros(eltype(VX), length(r), size(EToV, 1)) for _ in 1:3)
    for element in axes(EToV, 1)
        vertices = ntuple(i -> EToV[element, i], 4)
        for i in 1:4
            weight = barycentric[i] .* (2 .* barycentric[i] .- 1)
            x[:, element] .+= weight .* VX[vertices[i]]
            y[:, element] .+= weight .* VY[vertices[i]]
            z[:, element] .+= weight .* VZ[vertices[i]]
        end
        for i in 1:4, j in (i + 1):4
            node = edge_nodes[minmax(vertices[i], vertices[j])]
            weight = 4 .* barycentric[i] .* barycentric[j]
            x[:, element] .+= weight .* node[1]
            y[:, element] .+= weight .* node[2]
            z[:, element] .+= weight .* node[3]
        end
    end
    return x, y, z
end
