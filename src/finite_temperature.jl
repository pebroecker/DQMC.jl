function add_slice_sequence_left(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx])

    for slice in s.ranges[idx]
        slice_mat = slice_matrix(slice, s, p, l)
        curr_U = slice_mat * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
    s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
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
