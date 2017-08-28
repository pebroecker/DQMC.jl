function add_slice_sequence_left(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx])
    for slice in s.ranges[idx]
        slice_mat = slice_matrix(slice, s, p, l)
        curr_U = slice_mat * curr_U
    end
    s.u_stack[:, :, idx + 1], R, p = qr(curr_U, Val{true}; thin=true)
end

function add_slice_sequence_right(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack[:, :, idx + 1])

    for slice in reverse(s.ranges[idx])
        slice_mat = slice_matrix(slice, s, p, l)
        curr_U = transpose(slice_mat) * curr_U
    end
    s.u_stack[:, :, idx], R, p = qr(curr_U, Val{true}; thin=true)
end

function calculate_greens(s::stack_type, p::parameter_type, l::lattice)
    A = transpose(s.Ur) * s.Ul
    F = svdfact!(A)

    s.greens = eye(l.n_sites) - s.Ul * ctranspose(F[:Vt]) * spdiagm(1. ./ F[:S]) * ctranspose(F[:U]) * transpose(s.Ur)
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
                if diff > 1e-3
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
                if diff > 1e-3
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
