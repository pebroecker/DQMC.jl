@inline function onsite_update(s::stack_type, p::parameter_type, l::lattice)

    @inbounds for i in 1:l.n_sites
        gamma_up = exp(-1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        gamma_dn = exp(1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma_up * (1 - s.greens[i,i])) * (1 + gamma_dn * s.greens[i,i])

        if prob < 0
            println("Did you expect a sign problem?", prob)
        end

        if rand() < abs(prob) / (1. + abs(prob))
            u = -s.greens[:, i]
            u[i] += 1.
            s.greens -= kron(u * 1./(1 + gamma_up * u[i]), transpose(gamma_up * s.greens[i, :]))
            p.U_af_field[i, s.current_slice] *= -1.
        end
    end
end


function update(s::stack_type, p::parameter_type, l::lattice)
    s.n_interactions = 1
    s.curr_interaction = 1
    propagate(s, p, l)

    if s.direction == -1
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, -1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, 1)
        onsite_update(s, p, l)
    end

    if s.direction == 1
        onsite_update(s, p, l)
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, 1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, -1)
    end
end
