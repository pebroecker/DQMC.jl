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
  stack() = new()
end


function initialize_model_stack(s::stack_type, p::parameter_type, l::lattice)
    s.hopping_matrix[:] = 0.
    for r in 1:size(l.bonds_idx, 2)
        s.hopping_matrix[l.bonds_idx[1, r], l.bonds_idx[2, r]] = p.delta_tau
        s.hopping_matrix[l.bonds_idx[2, r], l.bonds_idx[1, r]] = p.delta_tau
    end
    # display(s.hopping_matrix)
    s.hopping_matrix_inv = expm(-1. * s.hopping_matrix)
    s.hopping_matrix = expm(s.hopping_matrix)
end


function get_wavefunction(s::stack_type, p::parameter_type, l::lattice)
    return eye(greens_type, l.n_sites)
end

function test_stack(s::stack_type, p::parameter_type, l::lattice)
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
    return spdiagm(exp(pref * p.lambda * p.af_field[:, slice]))
end


function get_hopping_matrix(s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    # M = eye(greens_type, l.n_sites, l.n_sites)
    #
    if pref > 0 return s.hopping_matrix
    else return s.hopping_matrix_inv    end
    # end
    #     for h in l.chkr_hop  M = h * M  end
    #     for h in reverse(l.chkr_hop)  M = h * M  end
    # else
    #     for h in l.chkr_hop_inv  M = h * M  end
    #     for h in reverse(l.chkr_hop_inv)   M = h * M   end
    # end
    #
    # return M
end

function slice_matrix(slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)

    # M = eye(greens_type, l.n_sites, l.n_sites)
    # println(get_interaction_matrix(p, l, slice))
    if pref > 0
        # return get_interaction_matrix(p, l, slice)
       return get_hopping_matrix(s, p, l) * get_interaction_matrix(p, l, slice)# * M
    #
    #     for h in l.chkr_hop    M = h * M    end
    #     for h in reverse(l.chkr_hop)    M = h * M    end
    else
        # return get_interaction_matrix(p, l, slice, -1.)
        # for h in l.chkr_hop_inv    M = h * M     end
        # for h in reverse(l.chkr_hop_inv)     M = h * M    end
        return get_interaction_matrix(p, l, slice, -1.) * get_hopping_matrix(s, p, l, -1.)
    end
    # return M
end
