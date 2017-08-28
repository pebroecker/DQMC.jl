type stack <: stack_type
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
  det::Float64
  greens_temp::Array{greens_type, 2}
  hopping_matrix::Array{greens_type, 2}
  hopping_matrix_inv::Array{greens_type, 2}
  free_fermion_wavefunction::Array{greens_type, 2}

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
  site_group_assignment::Array{Int64, 2}
  stack() = new()
end


function idx_to_rc(idx::Int64, L::Int64)
    r = Int(mod1(idx, L))
    c = Int(ceil(idx / L) - 1)
    return r, c
end

function rc_to_idx(r::Int64, c::Int64, L::Int64)
    return r + c * L
end

function get_site_environment(idx::Int64, L::Int64)
    indices = Int64[0, 0, idx, 0, 0]
    (r, c) = idx_to_rc(idx, L)
    indices[1] = rc_to_idx(mod1(r - 1, L), c, L)
    indices[2] = rc_to_idx(r, mod(c - 1 + L, L), L)
    indices[4] = rc_to_idx(r, mod(c + 1, L), L)
    indices[5] = rc_to_idx(mod1(r + 1, L), c, L)
    return indices
end

function initialize_model_stack(s::stack_type, p::parameter_type, l::lattice)
    K = zeros(greens_type, l.n_sites, l.n_sites)
    wave_hopping = zeros(greens_type, l.n_sites, l.n_sites)

    for i in 1:l.n_sites
        env = get_site_environment(i, l.L)
        K[i, env[1]] = p.delta_tau * p.t
        K[i, env[2]] = p.delta_tau * p.t
        K[i, env[4]] = p.delta_tau * p.t
        K[i, env[5]] = p.delta_tau * p.t
        wave_hopping[i, env[1]] = p.t * (1 + 0.1 * randn())
        wave_hopping[i, env[2]] = p.t * (1 + 0.1 * randn())
        wave_hopping[i, env[4]] = p.t * (1 + 0.1 * randn())
        wave_hopping[i, env[5]] = p.t * (1 + 0.1 * randn())
    end

    wave_hopping += wave_hopping'
    wave_hopping /= 2
    wave_eig = eigfact!(wave_hopping)
    println(wave_eig[:values])
    s.free_fermion_wavefunction = wave_eig[:vectors][:, 1:l.L * Int(l.L/2)]

    s.hopping_matrix = expm(Symmetric(K))
    s.hopping_matrix_inv = expm(Symmetric(-K))
end


function get_wavefunction(s::stack_type, p::parameter_type, l::lattice)
    if p.stack_handling == "ground_state"
        return s.free_fermion_wavefunction
    else
        return eye(greens_type, l.n_sites)
    end
end


function test_stack(s::stack_type, p::parameter_type, l::lattice)
    println(greens_type)
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

    println(greens_type)
    M = eye(greens_type, l.n_sites)
    M = multiply_slice_matrix_left(M, 1, s, p, l, 1.)
    println(typeof(M))
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

function get_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real=1)
    interaction_matrix = eye(greens_type, l.n_sites, l.n_sites)

    if prefactor < 0
        interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * interaction_matrix
    end

    if prefactor > 0
        interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, prefactor) * interaction_matrix
    end
    return interaction_matrix
end

function get_onsite_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real=1)
    return spdiagm(exp(prefactor * p.lambda * p.U_af_field[:, slice]))
end

function get_hopping_matrix(s::stack_type, p::parameter_type, l::lattice, pref::Real=1)
    if pref > 0 return s.hopping_matrix
    else return s.hopping_matrix_inv    end
end

function slice_matrix(slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Real=1)
    if pref > 0
        return get_hopping_matrix(s, p, l) * get_interaction_matrix(s, p, l, slice, pref)# * M
    else
        return get_interaction_matrix(s, p, l, slice, -1) * get_hopping_matrix(s, p, l, -1)
    end
end
