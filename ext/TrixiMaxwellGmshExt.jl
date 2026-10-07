module TrixiMaxwellGmshExt

using TrixiMaxwell: TrixiMaxwell, ImportedMesh
using Gmsh: gmsh

function TrixiMaxwell._read_gmsh(path::AbstractString; size_factor = 1.0, order = 1,
                                 verbose = false)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    isfile(path) || throw(ArgumentError("file $path does not exist"))
    initialized_here = !Bool(gmsh.isInitialized())
    initialized_here && gmsh.initialize()
    terminal = gmsh.option.getNumber("General.Terminal")
    gmsh.option.setNumber("General.Terminal", verbose ? 1 : 0)
    gmsh.open(path)
    try
        if isempty(gmsh.model.mesh.getElements(3)[1])
            # a geometry file: mesh it, scaling all prescribed sizes by size_factor
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", size_factor)
            gmsh.model.mesh.generate(3)
            order == 2 && gmsh.model.mesh.setOrder(2)
        elseif size_factor != 1.0 || order != 1
            throw(ArgumentError("size_factor and order apply to geometry files only, $path already contains a mesh"))
        end
        return ImportedMesh(gmsh_mesh_data(path))
    finally
        gmsh.model.remove()
        gmsh.option.setNumber("General.Terminal", terminal)
        initialized_here && gmsh.finalize()
    end
end

# Extract the current Gmsh model into plain arrays and dictionaries: corner
# vertices of all tetrahedra, physical tag per element, sorted vertex triples
# of all triangles in physical surface groups, and the mid-edge nodes of
# quadratic tetrahedra.
function gmsh_mesh_data(path)
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_index = Dict{Int, Int}(Int(tag) => i for (i, tag) in enumerate(node_tags))
    VX = coordinates[1:3:end]
    VY = coordinates[2:3:end]
    VZ = coordinates[3:3:end]

    EToV = Vector{NTuple{4, Int}}()
    element_groups = Int[]
    edge_nodes = Dict{NTuple{2, Int}, NTuple{3, Float64}}()
    for (_, entity) in gmsh.model.getEntities(3)
        physical = gmsh.model.getPhysicalGroupsForEntity(3, entity)
        length(physical) <= 1 ||
            throw(ArgumentError("volume $entity of $path belongs to $(length(physical)) physical groups, expected at most one"))
        group = isempty(physical) ? 0 : Int(first(physical))
        types, _, nodes = gmsh.model.mesh.getElements(3, entity)
        for (element_type, element_nodes) in zip(types, nodes)
            name, _, element_order, nodes_per_element, local_coordinates, _ = gmsh.model.mesh.getElementProperties(element_type)
            startswith(name, "Tetrahedron") ||
                throw(ArgumentError("$path contains $name elements, only tetrahedra are supported"))
            edges = element_order == 2 ? edge_vertex_pairs(local_coordinates) : ()
            for offset in 0:nodes_per_element:(length(element_nodes) - 1)
                vertices = ntuple(i -> node_index[Int(element_nodes[offset + i])], 4)
                push!(EToV, vertices)
                push!(element_groups, group)
                for (k, (i, j)) in enumerate(edges)
                    node = node_index[Int(element_nodes[offset + 4 + k])]
                    edge_nodes[minmax(vertices[i], vertices[j])] = (VX[node], VY[node],
                                                                    VZ[node])
                end
            end
        end
    end
    isempty(EToV) && throw(ArgumentError("$path contains no tetrahedra"))

    face_sets = Dict{Int, Vector{NTuple{3, Int}}}()
    for (_, entity) in gmsh.model.getEntities(2)
        physical = gmsh.model.getPhysicalGroupsForEntity(2, entity)
        isempty(physical) && continue
        types, _, nodes = gmsh.model.mesh.getElements(2, entity)
        for (element_type, element_nodes) in zip(types, nodes)
            name, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            startswith(name, "Triangle") ||
                throw(ArgumentError("$path contains $name surface elements, only triangles are supported"))
            for offset in 0:nodes_per_element:(length(element_nodes) - 1)
                triple = ntuple(i -> node_index[Int(element_nodes[offset + i])], 3)
                for tag in physical
                    push!(get!(face_sets, Int(tag), NTuple{3, Int}[]), triple)
                end
            end
        end
    end

    group_names = Dict{Int, String}()
    face_set_names = Dict{Int, String}()
    for (dim, tag) in gmsh.model.getPhysicalGroups()
        name = gmsh.model.getPhysicalName(dim, tag)
        isempty(name) && continue
        if dim == 3 && Int(tag) in element_groups
            group_names[Int(tag)] = name
        elseif dim == 2 && haskey(face_sets, Int(tag))
            face_set_names[Int(tag)] = name
        end
    end

    connectivity = permutedims(reduce(hcat, collect.(EToV)))
    return (; vertex_coordinates = (VX, VY, VZ), EToV = connectivity, element_groups,
            group_names, face_sets, face_set_names, edge_nodes)
end

# Vertex pair of each of the six mid-edge nodes of a quadratic tetrahedron, found
# from the reference coordinates of its ten nodes.
function edge_vertex_pairs(local_coordinates)
    reference = [local_coordinates[(3 * n - 2):(3 * n)] for n in 1:10]
    pairs = NTuple{2, Int}[]
    for n in 5:10
        k = findfirst(((i, j),) -> isapprox((reference[i] + reference[j]) / 2,
                                            reference[n]; atol = 1e-12),
                      [(i, j) for i in 1:4 for j in (i + 1):4])
        k === nothing &&
            throw(ArgumentError("node $n of a quadratic tetrahedron is not on an edge"))
        push!(pairs, [(i, j) for i in 1:4 for j in (i + 1):4][k])
    end
    return pairs
end

end # module
