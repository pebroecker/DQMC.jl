type stack <: stack_type
    u_stack_l::Array{greens_type, 3}
    d_stack_l::Array{real_type, 2}
    t_stack_l::Array{greens_type, 3}

    u_stack_r::Array{greens_type, 3}
    d_stack_r::Array{real_type, 2}
    t_stack_r::Array{greens_type, 3}

    u_temp::Array{greens_type, 2}
    d_temp::Array{real_type, 1}
    t_temp::Array{greens_type, 2}

    u_large::Array{greens_type, 2}
    d_large::Array{real_type, 1}
    t_large::Array{greens_type, 2}

    Ul::Array{greens_type, 2}
    Ur::Array{greens_type, 2}
    Dl::Array{real_type, 1}
    Dr::Array{real_type, 1}
    Tl::Array{greens_type, 2}
    Tr::Array{greens_type, 2}

    greens::Array{greens_type, 2}
    AB_greens::Array{greens_type, 2}
    det::Float64
    AB_det::Float64
    greens_temp::Array{greens_type, 2}
    AB_greens_temp::Array{greens_type, 2}
    free_fermion_wavefunction::Array{greens_type, 2}

    hopping_matrix::Array{greens_type, 2}
    hopping_matrix_inv::Array{greens_type, 2}
    hopping_matrix_AB::Array{greens_type, 2}
    hopping_matrix_AB_inv::Array{greens_type, 2}

    U::Array{greens_type, 2}
    Q::Array{greens_type, 2}
    D::Array{real_type, 1}
    R::Array{greens_type, 2}
    T::Array{greens_type, 2}

    replica::Int
    ranges::Array{UnitRange, 1}
    n_elements::Int
    current_slice::Int
    direction::Int
    n_interactions::Int
    curr_interaction::Int
    site_group_assignment::Array{Int64, 2}
    stack() = new()
end


function initialize_model_stack(s::stack_type, p::parameter_type, l::lattice)
    s.hopping_matrix = zeros(l.n_sites, l.n_sites)
    s.hopping_matrix_AB = zeros(p.N, p.N)
    s.hopping_matrix_AB_inv = zeros(p.N, p.N)
    wave_hopping = zeros(l.n_sites, l.n_sites)

    for r in 1:size(l.bonds_idx, 2)
        s.hopping_matrix[l.bonds_idx[1, r], l.bonds_idx[2, r]] = p.delta_tau * p.hoppings[l.bond_types[r]]
        s.hopping_matrix[l.bonds_idx[2, r], l.bonds_idx[1, r]] = p.delta_tau * p.hoppings[l.bond_types[r]]

        wave_hopping[l.bonds_idx[1, r], l.bonds_idx[2, r]] = p.hoppings[l.bond_types[r]] * (1 + 0.1 * randn())
        wave_hopping[l.bonds_idx[2, r], l.bonds_idx[1, r]] = p.hoppings[l.bond_types[r]] * (1 + 0.1 * randn())
    end
    s.hopping_matrix_inv = expm(-1. * s.hopping_matrix)
    s.hopping_matrix = expm(s.hopping_matrix)

    wave_hopping += wave_hopping'
    wave_hopping /= 2
    wave_eig = eigfact!(wave_hopping)
    # println(wave_eig[:values])
    # println(size(wave_eig[:vectors]))
    # display(s.hopping_matrix)
    if p.stack_handling == "ground_state"
        s.free_fermion_wavefunction = wave_eig[:vectors][:, 1:p.particles]
    else
        s.free_fermion_wavefunction =  eye(greens_type, l.n_sites)
    end

    enlarge(s.hopping_matrix, s.hopping_matrix_AB, 1, p, l)
    enlarge(s.hopping_matrix_inv, s.hopping_matrix_AB_inv, 1, p, l)
end

function get_wavefunction(s::stack_type, p::parameter_type, l::lattice)
    if p.stack_handling == "ground_state"
        return s.free_fermion_wavefunction
    else
        return eye(greens_type, l.n_sites)
    end
end

function test_stack(s::stack_type, p::parameter_type, l::lattice)
    M = eye(greens_type, l.n_sites)
    M = get_interaction_matrix(s, p, l, 1, 1.) * M
    M = M * get_interaction_matrix(s, p, l, 1, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Interaction matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = get_interaction_matrix(s, p, l, 1, -1.) * M
    M = M * get_interaction_matrix(s, p, l, 1, 1.)
    M -= eye(greens_type, l.n_sites)
    println("Interaction matrix test 2 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")

    M = eye(greens_type, l.n_sites)
    M = multiply_hopping_matrix_left(M, s, p, l, 1.)
    M = multiply_hopping_matrix_right(M, s, p, l, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Hopping matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = multiply_hopping_matrix_left(M, s, p, l, -1.)
    M = multiply_hopping_matrix_right(M, s, p, l, 1.)
    M -= eye(greens_type, l.n_sites)
    println("Hopping matrix test 2 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")


    M = eye(greens_type, l.n_sites)
    M = multiply_slice_matrix_left(M, 1, s, p, l, 1.)
    M = multiply_slice_matrix_right(M, 1, s, p, l, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = multiply_slice_matrix_left(M, 1, s, p, l, -1.)
    M = multiply_slice_matrix_right(M, 1, s, p, l, 1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 2 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    #
    # println("Building stack")
    # build_stack(s, p, l)
    # println("Recalculating greens")
    # println(size(s.Ul), " ", size(s.u_stack))
    # s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
    # s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    # s.Ur[:], s.Dr[:], s.Tr[:] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    # println("Calling calculate_greens")
    # calculate_greens(s, p, l)
    # println("Propagating greens")
    # for i in p.slices:-1:p.slices - p.safe_mult + 1
    #     s.greens[:] = multiply_slice_matrix_left(s.greens, i, s, p, l, -1.)
    #     s.greens[:] = multiply_slice_matrix_right(s.greens, i, s, p, l, 1.)
    # end
    # s.greens_temp[:] = s.greens[:]
    # s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    # idx = s.n_elements - 1
    # add_slice_sequence_right(s, idx, p, l)
    # s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    # println("Propagation test yields [$(minimum(abs(s.greens - s.greens_temp))), $(maximum(abs(s.greens - s.greens_temp)))]")
end

###############################################################################
# simple replica
###############################################################################

function get_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real)
    if prefactor < 0
        return get_onsite_interaction_matrix(s, p, l, slice, prefactor) * eye(greens_type, l.n_sites, l.n_sites)# * interaction_matrix
    else
        return get_onsite_interaction_matrix(s, p, l, slice, prefactor) * eye(greens_type, l.n_sites, l.n_sites)# * interaction_matrix
    end
end


function get_interaction_matrix_AB(s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real)
    interaction_matrix = eye(greens_type, p.N, p.N)

    if prefactor < 0
        interaction_matrix[1:l.n_sites, 1:l.n_sites] = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * eye(greens_type, l.n_sites)
    else
        interaction_matrix[1:l.n_sites, 1:l.n_sites] = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * eye(greens_type, l.n_sites)
    end
    return interaction_matrix
end


function get_onsite_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real)
    if s.replica == 1
        return spdiagm(exp(prefactor * p.lambda * p.U_af_field_A[:, slice] + prefactor * 0.05 * p.U))
    else
        return spdiagm(exp(prefactor * p.lambda * p.U_af_field_B[:, slice] + prefactor * 0.05 * p.U))
    end
end

function get_hopping_matrix(s::stack_type, p::parameter_type, l::lattice, pref::Real)
    if pref > 0 return s.hopping_matrix
    else return s.hopping_matrix_inv    end
end

function get_hopping_matrix_AB(s::stack_type, p::parameter_type, l::lattice, pref::Real)
    if pref > 0 return s.hopping_matrix_AB
    else return s.hopping_matrix_AB_inv    end
end

function slice_matrix(slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Real)
    if pref > 0
        return get_hopping_matrix(s, p, l, pref) * get_interaction_matrix(s, p, l, slice, pref)# * M
    else
        return get_interaction_matrix(s, p, l, slice, -1) * get_hopping_matrix(s, p, l, pref)
    end
end


###############################################################################
# combined replica
###############################################################################

# function get_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, replica::Int, slice::Int, prefactor::Real)
#     rep_interaction_matrix = eye(greens_type, p.N, p.N)
#     interaction_matrix = eye(greens_type, l.n_sites, l.n_sites)
#
#     if prefactor < 0
#         interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * interaction_matrix
#     end
#
#     if prefactor > 0
#         interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * interaction_matrix
#     end
#     enlarge(interaction_matrix, rep_interaction_matrix, s.replica, p, l)
#     return interaction_matrix
# end
#
# function get_hopping_matrix(replica::Int, s::stack_type, p::parameter_type, l::lattice, pref::Real=1)
#     if replica == 1
#         if pref > 0 return s.hopping_matrix_A
#         else return s.hopping_matrix_A_inv    end
#     else
#         if pref > 0 return s.hopping_matrix_B
#         else return s.hopping_matrix_B_inv    end
#     end
# end
#
# function slice_matrix_AB(replica::Int, slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Real)
#     if pref > 0
#         return get_hopping_matrix(replica, s, p, l) * get_interaction_matrix(s, p, l, replica, slice, pref)# * M
#     else
#         return get_interaction_matrix(s, p, l, replica, slice, -1) * get_hopping_matrix(replica, s, p, l, -1)
#     end
# end
