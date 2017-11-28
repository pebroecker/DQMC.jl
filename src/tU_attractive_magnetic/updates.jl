function update(s::stack_type, p::parameter_type, l::lattice)
    s.n_interactions = 1
    s.curr_interaction = 1
    @inbounds propagate(s, p, l)

    if s.direction == -1
        s.greens = get_interaction_matrix(p, l, s.current_slice, -1.) * s.greens * get_interaction_matrix(p, l, s.current_slice)
    end
    flipped = simple_update(s, p, l)
    if s.direction == 1
        s.greens = get_interaction_matrix(p, l, s.current_slice) * s.greens * get_interaction_matrix(p, l, s.current_slice, -1.)
    end
end

function simple_update(s::stack, p::parameters, l::lattice)
    spins_flipped = 0

    @inbounds for i in 1:l.n_sites
        gamma_up = exp(-1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        gamma_dn = exp(1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma_up * (1 - s.greens[i,i])) * (1 + gamma_dn * (1 - conj(s.greens[i,i])))
        # println("Probability is $(prob)")
        # if prob < 0
        #    println("Did you expect a sign problem?", prob)
        # end

        if rand() < abs(prob) / (1. + abs(prob))
            u = -s.greens[:, i]
            u[i] += 1.
            s.greens -= kron(u * 1./(1 + gamma_up * u[i]), transpose(gamma_up * s.greens[i, :]))
            p.U_af_field[i, s.current_slice] *= -1.
            spins_flipped += 1
        end
    end
    # println("Occupation\t", l.n_sites - sum(trace(s.greens)))
    return spins_flipped / l.n_sites
end
