"""
    read_gmsh(path; size_factor = 1.0, order = 1, verbose = false)

Read a tetrahedral mesh from a Gmsh `.msh` file of any format version into an
[`ImportedMesh`](@ref). Physical groups of volumes become element tags, physical
groups of surfaces become face sets, both keyed by the physical tag with the
physical name when one is defined. Quadratic tetrahedra keep their mid-edge
nodes, which makes the [`DGMultiMesh`](@ref) curved; higher orders are reduced to
their corner vertices. A `.geo` geometry file is meshed first, with every
prescribed mesh size multiplied by `size_factor`, which gives a resolution family
from one geometry, and at the given `order`, 1 or 2; Gmsh places the nodes of
order 2 on the curved geometry. Requires the Gmsh.jl package: `using Gmsh` before calling.
"""
read_gmsh(path::AbstractString; kwargs...) = _read_gmsh(path; kwargs...)

function _read_gmsh(path; kwargs...)
    error("read_gmsh requires the Gmsh.jl package, run `using Gmsh` before calling it")
end
