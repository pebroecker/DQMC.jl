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
    rel_phase = 1.

    @inbounds for i in 1:l.n_sites
        gamma = exp(-1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma * (1 - s.greens[i,i]))^2 / (gamma + 1)

        if prob < 0
            println("Did you expect a sign problem?", abs(imag(prob)))
            @printf "%.10e" abs(imag(prob))
        end

        if rand() < abs(prob) / (1. + abs(prob))
            rel_phase *= prob / abs(prob)
            u = -s.greens[:, i]
            u[i] += 1.
            s.greens -= kron(u * 1./(1 + gamma * u[i]), transpose(gamma * s.greens[i, :]))
            p.U_af_field[i, s.current_slice] *= -1.
            spins_flipped += 1
        end
    end
    return spins_flipped / l.n_sites, rel_phase
end
