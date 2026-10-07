"""
    FluxTotalFieldScatteredField(surface_flux, incident_field, is_total_field)

Surface flux for `DGSEM` on `P4estMesh` and `T8codeMesh` that applies the
total-field/scattered-field correction of [`TotalFieldScatteredField`](@ref) on
the interior faces between elements whose centroids `is_total_field(x)`
classifies differently, and behaves like `surface_flux` everywhere else. The
faces are marked on the first right-hand side evaluation and again after every
mesh adaptation; the interface must not cross mortars, so both sides of it have
to be refined equally.
"""
struct FluxTotalFieldScatteredField{Flux, IncidentField, Predicate, RealT <: Real}
    surface_flux::Flux
    incident_field::IncidentField
    is_total_field::Predicate
    signs::Vector{Int}
    coordinates::Vector{SVector{3, RealT}}
    time::Base.RefValue{RealT}
    mesh_signature::Base.RefValue{NTuple{3, Int}}
end

function FluxTotalFieldScatteredField(surface_flux, incident_field, is_total_field;
                                      RealT = Float64)
    return FluxTotalFieldScatteredField(surface_flux, incident_field, is_total_field,
                                        Int[], SVector{3, RealT}[], Ref(zero(RealT)),
                                        Ref((0, 0, 0)))
end

@inline function (flux::FluxTotalFieldScatteredField)(u_ll, u_rr, normal_direction,
                                                      equations)
    return flux.surface_flux(u_ll, u_rr, normal_direction, equations)
end

function Base.show(io::IO, flux::FluxTotalFieldScatteredField)
    print(io, "FluxTotalFieldScatteredField(", flux.surface_flux, ", ",
          flux.incident_field, ", ", count(!iszero, flux.signs), " interfaces)")
end

const DGSEMTotalFieldScatteredField = Trixi.DG{<:Trixi.LobattoLegendreBasis, <:Any,
                                               <:Trixi.SurfaceIntegralWeakForm{<:FluxTotalFieldScatteredField}}
const TFSFMesh = Union{Trixi.P4estMesh{3}, Trixi.T8codeMesh{3}}

function mark_interfaces!(flux::FluxTotalFieldScatteredField, mesh::TFSFMesh, dg::DGSEM,
                          cache)
    (; neighbor_ids, node_indices) = cache.interfaces
    (; node_coordinates) = cache.elements
    num_nodes = Trixi.nnodes(dg)
    index_range = eachnode(dg)
    num_interfaces = Trixi.ninterfaces(dg, cache)
    total_field = [flux.is_total_field(element_centroid(node_coordinates, element))
                   for element in Trixi.eachelement(dg, cache)]
    resize!(flux.signs, num_interfaces)
    fill!(flux.signs, 0)
    resize!(flux.coordinates, num_nodes^2 * num_interfaces)
    marked = 0
    for interface in Base.OneTo(num_interfaces)
        primary = neighbor_ids[1, interface]
        secondary = neighbor_ids[2, interface]
        total_field[primary] == total_field[secondary] && continue
        flux.signs[interface] = total_field[primary] ? 1 : -1
        marked += 1
        # walk the primary face like Trixi's interface loop
        primary_indices = node_indices[1, interface]
        i_start, i_step_i, i_step_j = Trixi.index_to_start_step_3d(primary_indices[1],
                                                                   index_range)
        j_start, j_step_i, j_step_j = Trixi.index_to_start_step_3d(primary_indices[2],
                                                                   index_range)
        k_start, k_step_i, k_step_j = Trixi.index_to_start_step_3d(primary_indices[3],
                                                                   index_range)
        i_primary, j_primary, k_primary = i_start, j_start, k_start
        for j in index_range
            for i in index_range
                flux.coordinates[tfsf_node_index(num_nodes, i, j, interface)] = SVector{3}(view(node_coordinates,
                                                                                                :,
                                                                                                i_primary,
                                                                                                j_primary,
                                                                                                k_primary,
                                                                                                primary))
                i_primary += i_step_i
                j_primary += j_step_i
                k_primary += k_step_i
            end
            i_primary += i_step_j
            j_primary += j_step_j
            k_primary += k_step_j
        end
    end
    marked == 0 &&
        throw(ArgumentError("is_total_field does not separate the mesh into two regions"))
    for mortar in Base.OneTo(Trixi.nmortars(cache.mortars))
        large = total_field[cache.mortars.neighbor_ids[5, mortar]]
        all(position -> total_field[cache.mortars.neighbor_ids[position, mortar]] == large,
            1:4) ||
            throw(ArgumentError("the total-field/scattered-field interface crosses a mortar; refine both sides equally"))
    end
    flux.mesh_signature[] = mesh_signature(dg, cache)
    return flux
end

# Adaptation changes the element, interface or mortar counts, so the marks are
# rebuilt whenever one of them differs.
function mesh_signature(dg, cache)
    return (Trixi.nelements(dg, cache), Trixi.ninterfaces(dg, cache),
            Trixi.nmortars(cache.mortars))
end

@inline function tfsf_node_index(num_nodes, i, j, interface)
    return (interface - 1) * num_nodes^2 + (j - 1) * num_nodes + i
end

# Mark the interfaces once and record the time for the incident field.
function Trixi.rhs_hyperbolic!(backend::Nothing, du, u, t, mesh::TFSFMesh,
                               equations::MaxwellEquations3D, boundary_conditions,
                               source_terms, dg::DGSEMTotalFieldScatteredField, cache)
    flux = dg.surface_integral.surface_flux
    flux.mesh_signature[] == mesh_signature(dg, cache) ||
        mark_interfaces!(flux, mesh, dg, cache)
    flux.time[] = t
    return invoke(Trixi.rhs_hyperbolic!,
                  Tuple{Nothing, Any, Any, Any,
                        Union{Trixi.TreeMesh{2}, Trixi.P4estMesh{2}, Trixi.P4estMeshView{2},
                              Trixi.T8codeMesh{2}, Trixi.TreeMesh{3}, Trixi.P4estMesh{3},
                              Trixi.T8codeMesh{3}}, Any, Any, Any, Trixi.DG, Any},
                  backend, du, u, t, mesh, equations, boundary_conditions, source_terms,
                  dg, cache)
end

# Both sides evaluate their own flux; on marked interfaces the neighbor trace is
# corrected by the incident field.
Base.@propagate_inbounds function tfsf_interface_flux!(surface_flux_values,
                                                       flux::FluxTotalFieldScatteredField,
                                                       equations, SolverT, u_interface,
                                                       interface_index, normal_direction,
                                                       primary_i_node_index,
                                                       primary_j_node_index,
                                                       primary_direction_index,
                                                       primary_element_index,
                                                       secondary_i_node_index,
                                                       secondary_j_node_index,
                                                       secondary_direction_index,
                                                       secondary_element_index)
    u_ll, u_rr = Trixi.get_surface_node_vars(u_interface, equations, SolverT,
                                             primary_i_node_index, primary_j_node_index,
                                             interface_index)
    sign = flux.signs[interface_index]
    if sign != 0
        num_nodes = size(u_interface, 3)
        x = flux.coordinates[tfsf_node_index(num_nodes, primary_i_node_index,
                                             primary_j_node_index, interface_index)]
        u_incident = flux.incident_field(x, flux.time[], equations)
        E_incident = sign * electric_field(u_incident)
        H_incident = sign * magnetic_field(u_incident)
        u_rr_seen = assemble(electric_field(u_rr) + E_incident,
                             magnetic_field(u_rr) + H_incident, u_rr, equations)
        u_ll_seen = assemble(electric_field(u_ll) - E_incident,
                             magnetic_field(u_ll) - H_incident, u_ll, equations)
    else
        u_rr_seen = u_rr
        u_ll_seen = u_ll
    end
    flux_primary = flux.surface_flux(u_ll, u_rr_seen, normal_direction, equations)
    flux_secondary = flux.surface_flux(u_rr, u_ll_seen, -normal_direction, equations)
    for v in Trixi.eachvariable(equations)
        surface_flux_values[v, primary_i_node_index, primary_j_node_index,
        primary_direction_index, primary_element_index] = flux_primary[v]
        surface_flux_values[v, secondary_i_node_index, secondary_j_node_index,
        secondary_direction_index, secondary_element_index] = flux_secondary[v]
    end
    return nothing
end

for Material in (Any, Heterogeneous)
    @eval Base.@propagate_inbounds function Trixi.calc_interface_flux!(surface_flux_values,
                                                                       ::Type{<:TFSFMesh},
                                                                       have_nonconservative_terms::Trixi.False,
                                                                       equations::MaxwellEquations3D{<:$Material},
                                                                       surface_integral::Trixi.SurfaceIntegralWeakForm{<:FluxTotalFieldScatteredField},
                                                                       SolverT::Type{<:Trixi.DG},
                                                                       u_interface,
                                                                       interface_index,
                                                                       normal_direction,
                                                                       primary_i_node_index,
                                                                       primary_j_node_index,
                                                                       primary_direction_index,
                                                                       primary_element_index,
                                                                       secondary_i_node_index,
                                                                       secondary_j_node_index,
                                                                       secondary_direction_index,
                                                                       secondary_element_index)
        return tfsf_interface_flux!(surface_flux_values, surface_integral.surface_flux,
                                    equations, SolverT, u_interface, interface_index,
                                    normal_direction, primary_i_node_index,
                                    primary_j_node_index, primary_direction_index,
                                    primary_element_index, secondary_i_node_index,
                                    secondary_j_node_index, secondary_direction_index,
                                    secondary_element_index)
    end
end
