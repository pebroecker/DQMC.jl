type stack <: stack_type
    free_fermion_wavefunction::Array{greens_type, 2}

    u_stack::Array{greens_type, 3}
    d_stack::Array{real_type, 2}
    t_stack::Array{greens_type, 3}

    Ul::Array{greens_type, 2}
    Ur::Array{greens_type, 2}
    Dl::Array{real_type, 1}
    Dr::Array{real_type, 1}
    Tl::Array{greens_type, 2}
    Tr::Array{greens_type, 2}

    greens::Array{greens_type, 2}
    greens_dn::Array{greens_type, 2}
    greens_temp::Array{greens_type, 2}
    hopping_matrix::Array{greens_type, 2}
    hopping_matrix_inv::Array{greens_type, 2}

    U::Array{greens_type, 2}
    Q::Array{greens_type, 2}
    D::Array{real_type, 1}
    R::Array{greens_type, 2}
    T::Array{greens_type, 2}

    ranges::Array{UnitRange, 1}
    n_elements::Int
    current_slice::Int
    direction::Int
    n_interactions::Int
    curr_interaction::Int

    det::real_type

    stack() = new()
end


function initialize_model_stack(s::stack_type, p::parameter_type, l::lattice)
    s.hopping_matrix[:] = 0.
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
    println(wave_eig[:values])
    println(size(wave_eig[:vectors]))
    # display(s.hopping_matrix)
    if p.stack_handling == "ground_state"
        s.free_fermion_wavefunction = wave_eig[:vectors][:, 1:p.particles]
    else
        s.free_fermion_wavefunction =  eye(greens_type, l.n_sites)
    end    
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
    M = multiply_slice_matrix_left(M, 1, s, p, l, 1.)
    # display(s.hopping_matrix[:, 1:8])
    # display(s.hopping_matrix[:, end - 8:end])
    M = multiply_slice_matrix_right(M, 1, s, p, l, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = multiply_slice_matrix_left(M, 1, s, p, l, -1.)
    M = multiply_slice_matrix_right(M, 1, s, p, l, 1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 2 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")

    build_stack(s, p, l)
    s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
    s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    s.Ur[:], s.Dr[:], s.Tr[:] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    calculate_greens(s, p, l)

    for i in p.slices:-1:p.slices - p.safe_mult + 1
        s.greens[:] = multiply_slice_matrix_left(s.greens, i, s, p, l, -1.)
        s.greens[:] = multiply_slice_matrix_right(s.greens, i, s, p, l, 1.)
    end

    s.greens_temp[:] = s.greens[:]
    s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    idx = s.n_elements - 1
    add_slice_sequence_right(s, idx, p, l)
    s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    println("Propagation test yields [$(minimum(abs(s.greens - s.greens_temp))), $(maximum(abs(s.greens - s.greens_temp)))]")
end

function get_interaction_matrix(p::parameter_type, l::lattice, slice::Int64, pref::Float64 = 1.)
    return spdiagm(exp(pref * (p.lambda * p.U_af_field[:, slice] - p.delta_tau * p.mu)))
end

function get_hopping_matrix(s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    if pref > 0 return s.hopping_matrix
    else return s.hopping_matrix_inv    end
end

function slice_matrix(slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    if pref > 0
       return get_hopping_matrix(s, p, l) * get_interaction_matrix(p, l, slice)# * M
    else
        return get_interaction_matrix(p, l, slice, -1.) * get_hopping_matrix(s, p, l, -1.)
    end
end
