@inline function onsite_update(s::stack_type, p::parameter_type, l::lattice)
    U_af_field = s.replica == 1 ? p.U_af_field_A : p.U_af_field_B

    @inbounds for i in 1:l.n_sites

        gamma = exp(-1. * 2 * U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma * (1 - s.greens[i,i]))^2 / (gamma + 1)

        # if abs(imag(prob)) > 1e-6
        #     println("Did you expect a sign problem?\t", abs(imag(prob)))
        #     # @printf "%.10e" abs(imag(prob))
        # end

        if rand() < abs(prob) / (1. + abs(prob))
            u = -s.greens[:, i]
            u[i] += 1.
            s.greens -= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * s.greens[i, :]))
            U_af_field[i, s.current_slice] *= -1.
        end
    end
end


@inline function onsite_update(s::stack_type, s_B::stack_type, p::parameter_type, l::lattice)

    U_af_field = s.replica == 1 ? p.U_af_field_A : p.U_af_field_B

    @inbounds for i in 1:l.n_sites

        gamma = exp(-1. * 2 * U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma * (1 - s.AB_greens[i,i]))^2 / (gamma + 1)

        if abs(imag(prob) / real(prob)) > 1e-6
            println("Did you expect a sign problem? $(s.current_slice) and $(i)\t$(abs(real(prob))) vs. $(abs(imag(prob))) - $(abs(prob) / (1. + abs(prob)))")
            # println(real(diag(s.AB_greens)))
            # display(real(s.AB_greens)); println("\n")
            # @printf "%.10e" abs(imag(prob))
        end

        if rand() < abs(prob) / (1. + abs(prob))
            u = -s.AB_greens[:, i]
            u[i] += 1.
            s.AB_greens -= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * s.AB_greens[i, :]))
            U_af_field[i, s.current_slice] *= -1.
        end
    end
end


function update(s::stack_type, p::parameter_type, l::lattice)
    s.n_interactions = 1
    s.curr_interaction = 1
    # println("In update\t", s.d_stack[:, end])
    propagate(s, p, l)
    # println("Propagate is done")
    if s.direction == -1
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, -1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, 1)
        onsite_update(s, p, l)
    end

    if s.direction == 1
        onsite_update(s, p, l)
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, 1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, -1)
    end
end


function update(s::stack_type, s_B::stack_type, p::parameter_type, l::lattice)
    s.n_interactions = 1
    s.curr_interaction = 1
    # println("In update\t", s.d_stack[:, end])
    propagate(s, s_B, p, l)
    # A = get_interaction_matrix_AB(s, p, l, s.current_slice, -1)
    # display(A'); println("\n")

    # println("Propagate is done")
    if s.direction == -1
        # println("interaction to the right $(s.current_slice)")
        s.AB_greens = get_interaction_matrix_AB(s, p, l, s.current_slice, -1) * s.AB_greens * get_interaction_matrix_AB(s, p, l, s.current_slice, 1)
        onsite_update(s, s_B, p, l)
    end

    if s.direction == 1
        # println("interaction up $(s.current_slice)")
        onsite_update(s, s_B, p, l)
        s.AB_greens = get_interaction_matrix_AB(s, p, l, s.current_slice, 1) * s.AB_greens * get_interaction_matrix_AB(s, p, l, s.current_slice, -1)
    end
end
