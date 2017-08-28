include("interaction_checkerboard.jl")

type stack <: stack_type
  u_stack::Array{greens_type, 3}
  d_stack::Array{real_type, 2}
  t_stack::Array{greens_type, 3}

  u_stack_dn::Array{greens_type, 3}
  d_stack_dn::Array{real_type, 2}
  t_stack_dn::Array{greens_type, 3}

  Ul::Array{greens_type, 2}
  Ur::Array{greens_type, 2}
  Dl::Array{real_type, 1}
  Dr::Array{real_type, 1}
  Tl::Array{greens_type, 2}
  Tr::Array{greens_type, 2}

  Ul_dn::Array{greens_type, 2}
  Ur_dn::Array{greens_type, 2}
  Dl_dn::Array{real_type, 1}
  Dr_dn::Array{real_type, 1}
  Tl_dn::Array{greens_type, 2}
  Tr_dn::Array{greens_type, 2}

  greens::Array{greens_type, 2}
  det::Float64
  greens_dn::Array{greens_type, 2}
  greens_temp::Array{greens_type, 2}
  greens_temp_dn::Array{greens_type, 2}
  hopping_matrix::Array{greens_type, 2}
  hopping_matrix_inv::Array{greens_type, 2}
  hopping_stencils::Dict{Int, Array{greens_type, 2}}
  hopping_stencils_inv::Dict{Int, Array{greens_type, 2}}
  hopping_vecs::Array{greens_type, 2}
  hopping_vals::Array{real_type, 1}

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

gam(aux_spin) = abs(aux_spin) == 1 ? 1 + sqrt(6) / 3 : 1 - sqrt(6) / 3
et(aux_spin) = abs(aux_spin) == 1 ? aux_spin * sqrt(2*(3 - sqrt(6))) : 0.5 * aux_spin * sqrt(2*(3 + sqrt(6)))


function build_stack(s::stack_type, p::parameter_type, l::lattice)
    println("Overwritten build_stack")
    s.u_stack[:, :, 1] = get_wavefunction(s, p, l)
    s.d_stack[:, 1] = ones(real_type, size(s.d_stack)[1])
    s.t_stack[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])

    s.u_stack_dn[:, :, 1] = get_wavefunction(s, p, l)
    s.d_stack_dn[:, 1] = ones(real_type, size(s.d_stack)[1])
    s.t_stack_dn[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])

    s.Ul_dn = copy(s.Ul)
    s.Ur_dn = copy(s.Ur)
    s.Dl_dn = copy(s.Dl)
    s.Dr_dn = copy(s.Dr)
    s.Tl_dn = copy(s.Tl)
    s.Tr_dn = copy(s.Tr)

    for i in 1:length(s.ranges)
        add_slice_sequence_left(s, i, p, l)
    end

    s.current_slice = p.slices + 1
    s.direction = -1
end



function initialize_model_stack(s::stack_type, p::parameter_type, l::lattice)

    s.u_stack_dn = copy(s.u_stack)
    s.d_stack_dn = copy(s.d_stack)
    s.t_stack_dn = copy(s.t_stack)
    s.greens_dn = copy(s.greens)
    s.greens_temp_dn = copy(s.greens_temp)

    K = zeros(greens_type, l.n_sites, l.n_sites)

    for i in 1:l.n_sites
        env = get_site_environment(i, l.L)
        K[i, env[1]] = p.delta_tau * p.t
        K[i, env[2]] = p.delta_tau * p.t# * exp(-p.phi / l.L * 1im)
        K[i, env[4]] = p.delta_tau * p.t# * exp(p.phi / l.L * 1im)
        K[i, env[5]] = p.delta_tau * p.t
    end

    s.hopping_matrix = expm(K)
    s.hopping_matrix_inv = expm(-K)

    s.hopping_stencils = Dict{Int, Array{greens_type, 2}}()
    s.hopping_stencils_inv = Dict{Int, Array{greens_type, 2}}()
    stencil = zeros(Float64, 5, 5)
    stencil[3, 1:2] = p.delta_tau * p.t
    stencil[3, 4:5] = p.delta_tau * p.t
    # stencil[3, 2] *= exp(-p.phi / l.L * 1im)
    # stencil[3, 4] *= exp(p.phi / l.L * 1im)
    stencil += stencil'

    hopping_stencil_eig = eigfact(stencil)
    s.hopping_vecs = hopping_stencil_eig[:vectors]
    s.hopping_vals = hopping_stencil_eig[:values]
    #hopping_stencils = Dict{Int, Array{Float64, 2}}()

    for aux_spin in [-2, -1, 0, 1, 2]
        s.hopping_stencils[aux_spin] = hopping_stencil_eig[:vectors] * diagm(exp(sqrt(p.delta_tau * p.W) * et(aux_spin) * hopping_stencil_eig[:values])) * hopping_stencil_eig[:vectors]'
        s.hopping_stencils_inv[aux_spin] = hopping_stencil_eig[:vectors] * diagm(exp(-sqrt(p.delta_tau * p.W) * et(aux_spin) * hopping_stencil_eig[:values])) * hopping_stencil_eig[:vectors]'
    end

    initialize_interaction_groups(s, p, l)
    println("Site group assignment is\n", s.site_group_assignment)
end

function get_wavefunction(s::stack_type, p::parameter_type, l::lattice)
    return eye(greens_type, l.n_sites)
end

function test_stack(s::stack_type, p::parameter_type, l::lattice)
    M = eye(greens_type, l.n_sites)
    M = get_interaction_matrix(s, p, l, 1, 1, 1.) * M
    M = M * get_interaction_matrix(s, p, l, 1, 1, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Interaction matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = get_interaction_matrix(s, p, l, 1, 1, -1.) * M
    M = M * get_interaction_matrix(s, p, l, 1, 1, 1.)
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
    M = multiply_slice_matrix_left(M, 1, s, p, l, 1, 1.)
    M = multiply_slice_matrix_right(M, 1, s, p, l, 1, -1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 1 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")
    M = eye(greens_type, l.n_sites)
    M = multiply_slice_matrix_left(M, 1, s, p, l, 1, -1.)
    M = multiply_slice_matrix_right(M, 1, s, p, l, 1, 1.)
    M -= eye(greens_type, l.n_sites)
    println("Slice matrix test 2 yields [$(minimum(abs(M))), $(maximum(abs(M)))]")

    M = slice_matrix(1, 1, s, p, l, 1)
    M = slice_matrix(-1, 1, s, p, l, 1)

    build_stack(s, p, l)
    s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
    s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    s.Ur[:], s.Dr[:], s.Tr[:] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))

    s.Ul_dn[:], s.Dl_dn[:], s.Tl_dn[:] = s.u_stack_dn[:, :, end], s.d_stack_dn[:, end], s.t_stack_dn[:, :, end]
    s.u_stack_dn[:, :, end], s.d_stack_dn[:, end], s.t_stack_dn[:, :, end] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))
    s.Ur_dn[:], s.Dr_dn[:], s.Tr_dn[:] = get_wavefunction(s, p, l), ones(size(s.d_stack, 1)), eye(greens_type, size(s.d_stack, 1), size(s.d_stack, 1))

    calculate_greens(s, p, l)
    # display(s.greens)
    # println()
    # display(s.greens_dn)
    # println()

    # println(diag(s.greens))
    # println(1 - diag(s.greens_dn))


    for i in p.slices:-1:p.slices - p.safe_mult + 1
        s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
        s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
        s.greens = get_onsite_interaction_matrix(s, p, l, i, 1, -1) * s.greens * get_onsite_interaction_matrix(s, p, l, i, 1, 1)
        # s.greens[:] = multiply_slice_matrix_left(s.greens, i, s, p, l, 1, -1.)
        # s.greens[:] = multiply_slice_matrix_right(s.greens, i, s, p, l, 1, 1.)
    end
    s.greens_temp[:] = s.greens[:]
    s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    idx = s.n_elements - 1
    add_slice_sequence_right(s, idx, p, l)
    s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end-1], s.d_stack[:, end-1], s.t_stack[:, :, end-1]
    calculate_greens(s, p, l)
    println("Propagation test yields [$(minimum(abs(s.greens - s.greens_temp))), $(maximum(abs(s.greens - s.greens_temp)))]")
end

function get_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, spin::Int, prefactor::Real)
    interaction_matrix = eye(Float64, l.n_sites, l.n_sites)

    if prefactor < 0
        interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, spin, prefactor) * interaction_matrix
    end

    if abs(p.W) > 1e-5
        group_order = prefactor > 0 ? collect(1:size(s.site_group_assignment, 1)) : reverse(collect(1:size(s.site_group_assignment, 1)))
        for group in group_order
            group_interaction_matrix = eye(Float64, l.n_sites, l.n_sites)
            interaction_matrix = get_group_interaction_matrix(group, s, p, l, slice, prefactor) * interaction_matrix
        end
    end

    if prefactor > 0
        interaction_matrix = get_onsite_interaction_matrix(s, p, l, slice, spin, prefactor) * interaction_matrix
    end
    return interaction_matrix
end

function get_group_interaction_matrix(group::Int, s::stack_type, p::parameter_type, l::lattice, slice::Int, prefactor::Real)
    interaction_matrix = eye(Float64, l.n_sites, l.n_sites)

    for site in s.site_group_assignment[group, :]
        if site == 0 continue end
        environment = get_site_environment(site, l.L)
        temp_stencil = prefactor > 0 ? s.hopping_stencils[p.W_af_field[site, slice]] : s.hopping_stencils_inv[p.W_af_field[site, slice]]

        for (ce, c) in enumerate(environment)
            for (re, r) in enumerate(environment)
                interaction_matrix[r, c] =  temp_stencil[re, ce]
            end
        end
    end
    return interaction_matrix
end

function get_onsite_interaction_matrix(s::stack_type, p::parameter_type, l::lattice, slice::Int, spin, prefactor::Real)
    return spdiagm(exp(spin * prefactor * p.lambda * p.U_af_field[:, slice]))
end


function get_hopping_matrix(s::stack_type, p::parameter_type, l::lattice, pref::Real=1)
    if pref > 0 return s.hopping_matrix
    else return s.hopping_matrix_inv    end
end


function slice_matrix(spin::Int, slice::Int, s::stack_type, p::parameter_type, l::lattice, pref::Real=1)
    if pref > 0
        return get_hopping_matrix(s, p, l) * get_interaction_matrix(s, p, l, slice, spin, 1)# * M
    else
        return get_interaction_matrix(s, p, l, slice, spin, -1) * get_hopping_matrix(s, p, l, -1)
    end
end


function add_slice_sequence_left(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx])

    for slice in s.ranges[idx]
        slice_mat = slice_matrix(1, slice, s, p, l)
        curr_U = slice_mat * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
    s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
    s.t_stack[:, :, idx + 1] = T * s.t_stack[:, :, idx]

    curr_U = copy(s.u_stack_dn[:, :, idx])

    for slice in s.ranges[idx]
        slice_mat = slice_matrix(-1, slice, s, p, l)
        curr_U = slice_mat * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack_dn[:, idx])
    s.u_stack_dn[:, :, idx + 1], s.d_stack_dn[:, idx + 1], T = decompose_udt(curr_U)
    s.t_stack_dn[:, :, idx + 1] = T * s.t_stack_dn[:, :, idx]
end


function add_slice_sequence_right(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx + 1])

    for slice in reverse(s.ranges[idx])
        slice_mat = slice_matrix(1, slice, s, p, l)
        curr_U = transpose(slice_mat) * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack[:, idx + 1])
    s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
    s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]

    curr_U = copy(s.u_stack_dn[:, :, idx + 1])

    for slice in reverse(s.ranges[idx])
        slice_mat = slice_matrix(-1, slice, s, p, l)
        curr_U = transpose(slice_mat) * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack_dn[:, idx + 1])
    s.u_stack_dn[:, :, idx], s.d_stack_dn[:, idx], T = decompose_udt(curr_U)
    s.t_stack_dn[:, :, idx] = T * s.t_stack_dn[:, :, idx + 1]
end

function calculate_greens(s::stack_type, p::parameter_type, l::lattice)
    A = spdiagm(s.Dl) * (s.Tl * transpose(s.Tr)) * spdiagm(s.Dr)

    M, S, N = decompose_udt(A)

    U = s.Ul * M
    D = S
    T = N * transpose(s.Ur)

    # A' * X = B
    # X = A'-1 * B
    # X' = B' * A^-1
    inside = ctranspose(\(ctranspose(T), U)) + diagm(D)
    Ui, Di, Ti = decompose_udt(inside)
    Di_inv = 1. ./ Di

    U_left = U * Ui
    T_left = spdiagm(Di) * Ti * T
    s.greens = \(T_left, ctranspose(U_left))

    ##########

    A = spdiagm(s.Dl_dn) * (s.Tl_dn * transpose(s.Tr_dn)) * spdiagm(s.Dr_dn)

    M, S, N = decompose_udt(A)

    U = s.Ul_dn * M
    D = S
    T = N * transpose(s.Ur_dn)

    inside = ctranspose(\(ctranspose(T), U)) + diagm(D)
    Ui, Di, Ti = decompose_udt(inside)
    Di_inv = 1. ./ Di

    U_left = U * Ui
    T_left = spdiagm(Di) * Ti * T
    s.greens_dn = \(T_left, ctranspose(U_left))
end
#
# function multiply_hopping_matrix_left(M::Array{greens_type, 2}, s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
#     return get_hopping_matrix(s, p, l, pref) * M
# end
#
# function multiply_hopping_matrix_right(M::Array{greens_type, 2},
#     s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
#     return M * get_hopping_matrix(s, p, l, pref)
# end

function multiply_slice_matrix_left(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, spin::Int, pref::Float64=1.)
    return slice_matrix(spin, slice, s, p, l, pref) * M
end

function multiply_slice_matrix_right(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, spin::Int, pref::Float64=1.)
    return M * slice_matrix(spin, slice, s, p, l, pref)
end


################################################################################
# Propagation
################################################################################
function propagate(s::stack_type, p::parameter_type, l::lattice)
    if s.direction == 1
        if mod(s.current_slice, p.safe_mult) == 0
            s.current_slice += 1
            if s.current_slice == 1
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
                s.Ur_dn[:], s.Dr_dn[:], s.Tr_dn[:] = s.u_stack_dn[:, :, 1], s.d_stack_dn[:, 1], s.t_stack_dn[:, :, 1]
                s.u_stack[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack[:, 1] = ones(size(s.d_stack)[1])
                s.t_stack[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])
                s.u_stack_dn[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack_dn[:, 1] = ones(size(s.d_stack)[1])
                s.t_stack_dn[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
                s.Ul_dn[:], s.Dl_dn[:], s.Tl_dn[:] = s.u_stack_dn[:, :, 1], s.d_stack_dn[:, 1], s.t_stack_dn[:, :, 1]
                calculate_greens(s, p, l)
                # println("from the bottom")
                # println(diag(s.greens) + diag(s.greens_dn) -1)
            elseif s.current_slice > 1 && s.current_slice < p.slices
                idx = Int((s.current_slice - 1) / p.safe_mult)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
                s.Ur_dn[:], s.Dr_dn[:], s.Tr_dn[:] = s.u_stack_dn[:, :, idx + 1], s.d_stack_dn[:, idx + 1], s.t_stack_dn[:, :, idx + 1]
                add_slice_sequence_left(s, idx, p, l)
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
                s.Ul_dn[:], s.Dl_dn[:], s.Tl_dn[:] = s.u_stack_dn[:, :, idx + 1], s.d_stack_dn[:, idx + 1], s.t_stack_dn[:, :, idx + 1]

                s.greens_temp[:] = s.greens[:]
                s.greens_temp[:] = multiply_hopping_matrix_right(s.greens_temp, s, p, l, -1.)
                s.greens_temp[:] = multiply_hopping_matrix_left(s.greens_temp, s, p, l, 1.)
                s.greens_temp_dn[:] = s.greens_dn[:]
                s.greens_temp_dn[:] = multiply_hopping_matrix_right(s.greens_temp_dn, s, p, l, -1.)
                s.greens_temp_dn[:] = multiply_hopping_matrix_left(s.greens_temp_dn, s, p, l, 1.)

                calculate_greens(s, p, l)
                # println(diag(s.greens) + diag(s.greens_dn) -1)

                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(1, " ", s.current_slice, "\t+1 Propagation stability\t", diff)
                end

                diff = maximum(diag(abs(s.greens_temp_dn - s.greens_dn)))
                if diff > 1e-4
                    println(-1, " ", s.current_slice, "\t+1 Propagation stability\t", diff)
                end
            else
                idx = s.n_elements - 1
                add_slice_sequence_left(s, idx, p, l)
                s.direction = -1
                s.current_slice = p.slices + 1
                propagate(s, p, l)
            end
        else
            s.current_slice += 1
            s.greens[:] = multiply_hopping_matrix_right(s.greens, s, p, l, -1.)
            s.greens[:] = multiply_hopping_matrix_left(s.greens, s, p, l, 1.)
            s.greens_dn[:] = multiply_hopping_matrix_right(s.greens_dn, s, p, l, -1.)
            s.greens_dn[:] = multiply_hopping_matrix_left(s.greens_dn, s, p, l, 1.)
        end
    elseif s.direction == -1
        if mod(s.current_slice - 1, p.safe_mult) == 0
            s.current_slice -= 1
            idx = Int(s.current_slice / p.safe_mult) + 1
            if s.current_slice == p.slices
                # println("Recalculating greens from the top")
                s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
                s.Ul_dn[:, :], s.Dl_dn[:], s.Tl_dn[:, :] = s.u_stack_dn[:, :, end], s.d_stack_dn[:, end], s.t_stack_dn[:, :, end]
                s.u_stack[:, :, end] = get_wavefunction(s, p, l)
                s.d_stack[:, end] = ones(p.particles)
                s.t_stack[:, :, end] = eye(greens_type, p.particles, p.particles)
                s.u_stack_dn[:, :, end] = get_wavefunction(s, p, l)
                s.d_stack_dn[:, end] = ones(p.particles)
                s.t_stack_dn[:, :, end] = eye(greens_type, p.particles, p.particles)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
                s.Ur_dn[:], s.Dr_dn[:], s.Tr_dn[:] = s.u_stack_dn[:, :, end], s.d_stack_dn[:, end], s.t_stack_dn[:, :, end]
                calculate_greens(s, p, l)
                # println(1 - diag(s.greens + s.greens_dn))

                # println("Propagating hopping from the top")
                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
                s.greens_dn = multiply_hopping_matrix_left(s.greens_dn, s, p, l, -1.)
                s.greens_dn = multiply_hopping_matrix_right(s.greens_dn, s, p, l, 1.)

            elseif s.current_slice > 0 && s.current_slice < p.slices
                s.greens_temp[:] = s.greens[:]
                s.greens_temp_dn[:] = s.greens_dn[:]
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
                s.Ul_dn[:], s.Dl_dn[:], s.Tl_dn[:] = s.u_stack_dn[:, :, idx], s.d_stack_dn[:, idx], s.t_stack_dn[:, :, idx]
                add_slice_sequence_right(s, idx, p, l)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
                s.Ur_dn[:], s.Dr_dn[:], s.Tr_dn[:] = s.u_stack_dn[:, :, idx], s.d_stack_dn[:, idx], s.t_stack_dn[:, :, idx]
                calculate_greens(s, p, l)
                # println(1 - diag(s.greens + s.greens_dn))

                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(1, " ", s.current_slice, "\t-1  Propagation stability\t", diff)
                end
                diff = maximum(diag(abs(s.greens_temp_dn - s.greens_dn)))
                if diff > 1e-4
                    println(-1, " ", s.current_slice, "\t-1  Propagation stability\t", diff)
                end

                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
                s.greens_dn = multiply_hopping_matrix_left(s.greens_dn, s, p, l, -1.)
                s.greens_dn = multiply_hopping_matrix_right(s.greens_dn, s, p, l, 1.)
            elseif s.current_slice == 0
                add_slice_sequence_right(s, 1, p, l)
                s.direction = 1
                propagate(s, p, l)
            end
        else
            s.current_slice -= 1

            s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
            s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
            s.greens_dn = multiply_hopping_matrix_left(s.greens_dn, s, p, l, -1.)
            s.greens_dn = multiply_hopping_matrix_right(s.greens_dn, s, p, l, 1.)
        end
    end
end
