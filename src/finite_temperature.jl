function add_slice_sequence_left(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx])

    for slice in s.ranges[idx]
        slice_mat = slice_matrix(slice, s, p, l)
        curr_U = slice_mat * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
    s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
    # display(s.d_stack[:, idx + 1])
    s.t_stack[:, :, idx + 1] = T * s.t_stack[:, :, idx]
end

function add_slice_sequence_right(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx + 1])

    for slice in reverse(s.ranges[idx])
        slice_mat = slice_matrix(slice, s, p, l)
        curr_U = transpose(slice_mat) * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack[:, idx + 1])
    s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
    s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
end


function calculate_greens(s::stack_type, p::parameter_type, l::lattice)
    large_M = zeros(2 * l.n_sites, 2 * l.n_sites)

    Ul = s.Ul
    Dl = s.Dl
    Tl = s.Tl

    Ur = transpose(s.Tr)
    Dr = s.Dr
    Tr = transpose(s.Ur)

    a = 1:l.n_sites
    b = l.n_sites + 1:2 * l.n_sites

    large_M[a, a] = eye(l.n_sites) / (Tr * Ul)
    large_M[b, b] = eye(l.n_sites) / (Tl * Ur)
    large_M[b, a] = -diagm(Dr)
    large_M[a, b] = diagm(Dl)

    large_U, large_D, large_T = decompose_udt(large_M)

    large_M[:] = 0
    large_M[a, a] = Ul
    large_M[b, b] = Ur
    U_p = large_M * large_U

    large_M[:] = 0
    large_M[a, a] = Tr
    large_M[b, b] = Tl
    T_p = large_T * large_M

    large_M[:] = spdiagm(1./large_D) * ctranspose(U_p)
    large_G = \(T_p, large_M)
    s.greens = large_G[a, a]
    s.det = sum(log(large_D))
end


function calculate_greens_old(s::stack_type, p::parameter_type, l::lattice)
    A = spdiagm(s.Dl) * (s.Tl * transpose(s.Tr)) * spdiagm(s.Dr)

    M, S, N = decompose_udt(A)

    U = s.Ul * M
    D = S
    T = N * transpose(s.Ur)

    # println("----------------")
    # println(s.Dl[1:5], " - ", s.Dl[end - 5:end])
    # println(s.Dr[1:5])
    # println(s.Dr[end - 5:end])
    # A' * X = B
    # X = A'-1 * B
    # X' = B' * A^-1
    inside = ctranspose(\(ctranspose(T), U)) + diagm(D)
    Ui, Di, Ti = decompose_udt(inside)
    Di_inv = 1. ./ Di

    U_left = U * Ui
    T_left = spdiagm(Di) * Ti * T

    # G = U^{-1} * T^{-1}
    s.greens = \(T_left, ctranspose(U_left))

    # For the determinant:
    # N is permuted, upper triangular from UDT => +/- 1
    # s.Ur is unitary matrix => +/- 1
    # Ti is permuted, upper triangular from UDT => +/- 1
    # s.Ul is unitary matrix => +/- 1
    # M is unitary matrix => +/- 1
    # Ui is unitary matrix => +/- 1
    s.det = sum(log(Di))
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
                s.u_stack[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack[:, 1] = ones(size(s.d_stack)[1])
                s.t_stack[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
                calculate_greens(s, p, l)
            elseif s.current_slice > 1 && s.current_slice < p.slices
                idx = Int((s.current_slice - 1) / p.safe_mult)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
                add_slice_sequence_left(s, idx, p, l)
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]

                s.greens_temp[:] = s.greens[:]
                s.greens_temp[:] = multiply_hopping_matrix_right(s.greens_temp, s, p, l, -1.)
                s.greens_temp[:] = multiply_hopping_matrix_left(s.greens_temp, s, p, l, 1.)

                calculate_greens(s, p, l)
                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t+1 Propagation stability\t", diff)
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
            if s.curr_interaction == s.n_interactions
                s.greens[:] = multiply_hopping_matrix_right(s.greens, s, p, l, -1.)
                s.greens[:] = multiply_hopping_matrix_left(s.greens, s, p, l, 1.)
            end
        end
    elseif s.direction == -1
        if mod(s.current_slice - 1, p.safe_mult) == 0
            s.current_slice -= 1
            idx = Int(s.current_slice / p.safe_mult) + 1
            if s.current_slice == p.slices
                s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
                s.u_stack[:, :, end] = get_wavefunction(s, p, l)
                s.d_stack[:, end] = ones(p.particles)
                s.t_stack[:, :, end] = eye(greens_type, p.particles, p.particles)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
                calculate_greens(s, p, l)

                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)

            elseif s.current_slice > 0 && s.current_slice < p.slices
                s.greens_temp[:] = s.greens[:]

                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
                add_slice_sequence_right(s, idx, p, l)
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
                calculate_greens(s, p, l)
                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t-1  Propagation stability\t", diff)
                end

                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
            elseif s.current_slice == 0
                add_slice_sequence_right(s, 1, p, l)
                s.direction = 1
                propagate(s, p, l)
            end
        else
            s.current_slice -= 1
            if s.curr_interaction == 1
                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
            end
        end
    end
end
