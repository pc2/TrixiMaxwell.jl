# DGSEM evaluates one numerical flux per interface node and writes it to both
# neighbours. With a material jump the two sides need their own flux, so the
# surface flux is evaluated once per side as on DGMulti.

Base.@propagate_inbounds function Trixi.calc_interface_flux!(surface_flux_values,
                                                             ::Type{<:Union{Trixi.P4estMesh{3},
                                                                            Trixi.T8codeMesh{3}}},
                                                             have_nonconservative_terms::Trixi.False,
                                                             equations::MaxwellEquations3D{Heterogeneous},
                                                             surface_integral,
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
    (; surface_flux) = surface_integral
    u_ll, u_rr = Trixi.get_surface_node_vars(u_interface, equations, SolverT,
                                             primary_i_node_index, primary_j_node_index,
                                             interface_index)
    flux_primary = surface_flux(u_ll, u_rr, normal_direction, equations)
    flux_secondary = surface_flux(u_rr, u_ll, -normal_direction, equations)
    for v in Trixi.eachvariable(equations)
        surface_flux_values[v, primary_i_node_index, primary_j_node_index,
        primary_direction_index, primary_element_index] = flux_primary[v]
        surface_flux_values[v, secondary_i_node_index, secondary_j_node_index,
        secondary_direction_index, secondary_element_index] = flux_secondary[v]
    end
    return nothing
end

function Trixi.calc_interface_flux!(backend::Nothing, surface_flux_values,
                                    mesh::Trixi.TreeMesh{3},
                                    have_nonconservative_terms::Trixi.False,
                                    equations::MaxwellEquations3D{Heterogeneous},
                                    surface_integral, dg::Trixi.DG, cache)
    (; surface_flux) = surface_integral
    (; u, neighbor_ids, orientations) = cache.interfaces

    Trixi.@threaded for interface in Trixi.eachinterface(dg, cache)
        left_id = neighbor_ids[1, interface]
        right_id = neighbor_ids[2, interface]
        left_direction = 2 * orientations[interface]
        right_direction = 2 * orientations[interface] - 1
        for j in eachnode(dg), i in eachnode(dg)
            u_ll, u_rr = Trixi.get_surface_node_vars(u, equations, dg, i, j, interface)
            flux_left = surface_flux(u_ll, u_rr, orientations[interface], equations)
            # the right element sees the interface through the opposite normal
            flux_right = -surface_flux(u_rr, u_ll,
                                       -unit_normal(orientations[interface],
                                                    eltype(u_ll)),
                                       equations)
            for v in Trixi.eachvariable(equations)
                surface_flux_values[v, i, j, left_direction, left_id] = flux_left[v]
                surface_flux_values[v, i, j, right_direction, right_id] = flux_right[v]
            end
        end
    end
    return nothing
end

# Mortars: the large element receives the negated secondary flux, so its own
# flux through its outward normal is stored with a flipped sign.
Base.@propagate_inbounds function Trixi.calc_mortar_flux!(fstar_primary, fstar_secondary,
                                                          mesh::Union{Trixi.P4estMesh{3},
                                                                      Trixi.T8codeMesh{3}},
                                                          have_nonconservative_terms::Trixi.False,
                                                          equations::MaxwellEquations3D{Heterogeneous},
                                                          surface_integral, dg::Trixi.DG,
                                                          cache, mortar_index,
                                                          position_index, normal_direction,
                                                          i_node_index, j_node_index)
    (; u) = cache.mortars
    (; surface_flux) = surface_integral
    u_ll, u_rr = Trixi.get_surface_node_vars(u, equations, dg, position_index,
                                             i_node_index, j_node_index, mortar_index)
    flux_small = surface_flux(u_ll, u_rr, normal_direction, equations)
    flux_large = surface_flux(u_rr, u_ll, -normal_direction, equations)
    Trixi.set_node_vars!(fstar_primary, flux_small, equations, dg, i_node_index,
                         j_node_index, position_index)
    Trixi.set_node_vars!(fstar_secondary, -flux_large, equations, dg, i_node_index,
                         j_node_index, position_index)
    return nothing
end
