# Points, cells and cell data of an imported mesh in the layout WriteVTK expects:
# all tetrahedra with their element group, and all tagged faces with their tag and
# a flag whether the face lies on the domain boundary.
function mesh_vtk_data(imported::ImportedMesh)
    VX, VY, VZ = imported.vertex_coordinates
    points = permutedims(hcat(VX, VY, VZ))
    EToV = imported.EToV

    volume_cells = [MeshCell(VTKCellTypes.VTK_TETRA, EToV[e, :]) for e in axes(EToV, 1)]

    fv = StartUpDG.face_vertices(StartUpDG.Tet())
    FToF = StartUpDG.connect_mesh(EToV, fv)
    lookup = face_lookup(EToV, fv)

    face_cells = MeshCell{VTKCellTypes.VTKCellType, Vector{Int}}[]
    face_tags = Int[]
    face_on_boundary = Int[]
    for tag in sort(collect(keys(imported.face_sets))), triple in imported.face_sets[tag]
        push!(face_cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, collect(triple)))
        push!(face_tags, tag)
        face_id = lookup[triple]
        push!(face_on_boundary, FToF[face_id] == face_id ? 1 : 0)
    end

    return (; points, volume_cells, element_groups = imported.element_groups,
            face_cells, face_tags, face_on_boundary)
end

"""
    write_mesh_vtk(imported::ImportedMesh, prefix)

Write `prefix_volume.vtu` with all tetrahedra and the cell data `element_group`,
and `prefix_faces.vtu` with all tagged faces and the cell data `face_tag` and
`on_boundary`, for inspection in ParaView. Returns the written file names.
"""
function write_mesh_vtk(imported::ImportedMesh, prefix::AbstractString)
    data = mesh_vtk_data(imported)
    files = String[]

    volume = vtk_grid(prefix * "_volume", data.points, data.volume_cells)
    volume["element_group", VTKCellData()] = data.element_groups
    append!(files, vtk_save(volume))

    if !isempty(data.face_cells)
        faces = vtk_grid(prefix * "_faces", data.points, data.face_cells)
        faces["face_tag", VTKCellData()] = data.face_tags
        faces["on_boundary", VTKCellData()] = data.face_on_boundary
        append!(files, vtk_save(faces))
    end

    return files
end

"""
    write_solution_vtk(u_ode, semi, filename; solution_variables = cons2cons)

Write the solution `u_ode` of the DGMulti semidiscretization `semi` as
Lagrange tetrahedra of the solution degree to `filename.vtu`, one point field per variable
of `solution_variables`. Returns the written file names.
"""
function write_solution_vtk(u_ode, semi, filename::AbstractString;
                            solution_variables = Trixi.cons2cons)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array(u_ode, semi)
    converted = map(u_node -> solution_variables(u_node, equations), u)
    names = collect(String, Trixi.varnames(solution_variables, equations))
    data = [Trixi.get_component(converted, v) for v in eachindex(names)]
    return StartUpDG.export_to_vtk(plotting_basis(solver.basis), mesh.md, data, names,
                                   filename)
end

# Lagrange cells of the solution degree represent the solution exactly; the
# StartUpDG default plots at degree 10.
function plotting_basis(rd)
    rd.Nplot == rd.N && return rd
    rd_plot = StartUpDG.RefElemData(rd.element_type, rd.N; Nplot = rd.N)
    all(map((a, b) -> a ≈ b, rd_plot.rst, rd.rst)) || return rd
    return rd_plot
end

"""
    write_solution_vtk(u_odes, times, semi, prefix; solution_variables = cons2cons)

Write one `.vtu` file per entry of `u_odes` and a ParaView collection
`prefix.pvd` that attaches `times` to them, for example
`write_solution_vtk(sol.u, sol.t, semi, "cavity")`. Returns the collection file
name.
"""
function write_solution_vtk(u_odes::AbstractVector, times::AbstractVector, semi,
                            prefix::AbstractString; kwargs...)
    length(u_odes) == length(times) ||
        throw(ArgumentError("got $(length(u_odes)) solutions and $(length(times)) times"))
    directory = dirname(prefix)
    isempty(directory) || mkpath(directory)
    entries = Tuple{Float64, String}[]
    for (i, (u_ode, t)) in enumerate(zip(u_odes, times))
        step = prefix * "_" * lpad(i, 6, '0')
        files = write_solution_vtk(u_ode, semi, step; kwargs...)
        push!(entries, (Float64(t), basename(first(files))))
    end
    collection = prefix * ".pvd"
    write_pvd(collection, entries)
    return collection
end

# ParaView collection listing `(time, file)` pairs; file names are relative to the
# collection, which WriteVTK only supports for files it has not saved yet.
function write_pvd(path::AbstractString, entries)
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io,
                "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (t, file) in entries
            println(io, "    <DataSet timestep=\"", t, "\" part=\"0\" file=\"", file,
                    "\"/>")
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
    return path
end
