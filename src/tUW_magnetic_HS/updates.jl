@inline function onsite_update(s::stack_type, p::parameter_type, l::lattice)
    @inbounds for i in 1:l.n_sites
        gamma_up = exp(-1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        gamma_dn = exp(1. * 2 * p.U_af_field[i, s.current_slice] * p.lambda) - 1
        prob = (1 + gamma_up * (1 - s.greens[i,i])) * (1 + gamma_dn * (1 - s.greens_dn[i,i]))

        if prob < 0
            println("Did you expect a sign problem?", prob)
        end

        # println("Flipping with $(abs(prob) / (1. + abs(prob)))")

        if rand() < abs(prob) / (1. + abs(prob))
            u = -s.greens[:, i]
            u[i] += 1.
            s.greens -= kron(u * 1./(1 + gamma_up * u[i]), transpose(gamma_up * s.greens[i, :]))

            u = -s.greens_dn[:, i]
            u[i] += 1.
            s.greens_dn -= kron(u * 1./(1 + gamma_dn * u[i]), transpose(gamma_dn * s.greens_dn[i, :]))

            p.U_af_field[i, s.current_slice] *= -1.
        end
    end
end


function update(s::stack_type, p::parameter_type, l::lattice)
    propagate(s, p, l)
    s.n_interactions = size(s.site_group_assignment, 1)

    group_order = (s.direction == 1) ? collect(1:s.n_interactions) : collect(reverse(1:s.n_interactions))

    # println(1 - diag(s.greens + s.greens_dn))

    if s.direction == -1
        # println("Propagating onsite interaction")
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, 1, -1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, 1, 1)
        s.greens_dn = get_onsite_interaction_matrix(s, p, l, s.current_slice, -1, -1) * s.greens_dn * get_onsite_interaction_matrix(s, p, l, s.current_slice, -1, 1)
        onsite_update(s, p, l)
    end
    # println(1 - diag(s.greens + s.greens_dn))

    for group in group_order
        if s.direction == -1
            s.greens = get_group_interaction_matrix(group, s, p, l, s.current_slice, -1) * s.greens * get_group_interaction_matrix(group, s, p, l, s.current_slice, 1)
            s.greens_dn = get_group_interaction_matrix(group, s, p, l, s.current_slice, -1) * s.greens_dn * get_group_interaction_matrix(group, s, p, l, s.current_slice, 1)
        end
        s.curr_interaction = group

        for site in s.site_group_assignment[group, :]
            if p.W < 1e-5 continue end
            if site == 0 continue end

            # println("Attemping update on site $(site) / $(l.n_sites) | slice $(s.current_slice) / $(p.slices)")

            propose_spin = rand([-2, -1, 1, 2])
            Delta_spin = et(propose_spin) - et(p.W_af_field[site, s.current_slice])
            ratio_prefactor = gam(propose_spin) / gam(p.W_af_field[site, s.current_slice])

            update_U = s.hopping_vecs * diagm(exp(sqrt(p.delta_tau * p.W) * Delta_spin * s.hopping_vals) - ones(5))
            update_V = s.hopping_vecs'
            u_prime = zeros(Complex{Float64}, l.n_sites, 5)
            v = zeros(Complex{Float64}, 5, l.n_sites)
            environment = get_site_environment(site, l.L)
            for (m, n) in enumerate(environment)
                u_prime[n, :] = update_U[m, :]
                v[:, n] = update_V[:, m]
            end
            u = (eye(l.n_sites) - s.greens) * u_prime
            u_dn = (eye(l.n_sites) - s.greens_dn) * u_prime

            update_ratio = det(eye(l.n_sites) + u * v) * det(eye(l.n_sites) + u_dn * v)
            r = update_ratio * ratio_prefactor

            if abs(imag(r)/real(r)) > 1e-6 println("Did you expect a sign problem? $(imag(r) / real(r))") end
            # println("Changing with $(abs(r) / (1. + abs(r)))")

            if rand() < real(r / (1 + r))
                # display(- (u * inv(eye(5) + v * u)) * (v * s.greens))
                p.W_af_field[site, s.current_slice] = propose_spin
                s.greens = s.greens - (u * inv(eye(5) + v * u)) * (v * s.greens)
                s.greens_dn = s.greens_dn - (u_dn * inv(eye(5) + v * u_dn)) * (v * s.greens_dn)
            end
        end

        if s.direction == 1
            s.greens = get_group_interaction_matrix(group, s, p, l, s.current_slice, 1) * s.greens * get_group_interaction_matrix(group, s, p, l, s.current_slice, -1)
            s.greens_dn = get_group_interaction_matrix(group, s, p, l, s.current_slice, 1) * s.greens_dn * get_group_interaction_matrix(group, s, p, l, s.current_slice, -1)
        end
    end

    rel_phase = 1.

    if s.direction == 1
        onsite_update(s, p, l)
        s.greens = get_onsite_interaction_matrix(s, p, l, s.current_slice, 1, 1) * s.greens * get_onsite_interaction_matrix(s, p, l, s.current_slice, 1, -1)
        s.greens_dn = get_onsite_interaction_matrix(s, p, l, s.current_slice, -1, 1) * s.greens_dn * get_onsite_interaction_matrix(s, p, l, s.current_slice, -1, -1)
    end
end

@inline function explicit_det(M::Array{greens_type, 2})
    d = 0.

    # column 1
    sd = 0.
    ssd = 0.
    ssd += M[3, 3] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 3] + M[3, 5] * M[4, 3] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 3] + M[3, 4] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 4]
    sd += M[2, 2] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [3, 4, 5]]))

    ssd = 0.
    ssd += M[3, 2] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 2] + M[3, 5] * M[4, 2] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 2] + M[3, 4] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 4]
    sd -= M[2, 3] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [2, 4, 5]]))

    ssd = 0.
    ssd += M[3, 2] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 2] + M[3, 5] * M[4, 2] * M[5, 3]
    ssd -= M[3, 5] * M[4, 3] * M[5, 2] + M[3, 3] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 3]
    sd += M[2, 4] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [2, 3, 5]]))

    ssd = 0.
    ssd += M[3, 2] * M[4, 3] * M[5, 4] + M[3, 3] * M[4, 4] * M[5, 2] + M[3, 4] * M[4, 2] * M[5, 3]
    ssd -= M[3, 4] * M[4, 3] * M[5, 2] + M[3, 3] * M[4, 2] * M[5, 4] + M[3, 2] * M[4, 4] * M[5, 3]
    sd -= M[2, 5] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [2, 3, 4]]))

    d += M[1, 1] * sd

    # column 2
    sd = 0.
    ssd = 0.
    ssd += M[3, 3] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 3] + M[3, 5] * M[4, 3] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 3] + M[3, 4] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 4]
    sd += M[2, 1] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [3, 4, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 4]
    sd -= M[2, 3] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 4, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 3]
    ssd -= M[3, 5] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 3]
    sd += M[2, 4] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 3, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 3] * M[5, 4] + M[3, 3] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 3]
    ssd -= M[3, 4] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 4] + M[3, 1] * M[4, 4] * M[5, 3]
    sd -= M[2, 5] * ssd
    d -= M[1, 2] * sd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 3, 4]]))

    # column 3
    sd = 0.
    ssd = 0.
    ssd += M[3, 2] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 2] + M[3, 5] * M[4, 2] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 2] + M[3, 4] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 4]
    sd += M[2, 1] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [2, 4, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 4] * M[5, 5] + M[3, 4] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 4]
    ssd -= M[3, 5] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 4]
    sd -= M[2, 2] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 4, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 2]
    ssd -= M[3, 5] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 2]
    sd += M[2, 4] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 2, 5]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 4] + M[3, 2] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 2]
    ssd -= M[3, 4] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 4] + M[3, 1] * M[4, 4] * M[5, 2]
    sd -= M[2, 5] * ssd
    d += M[1, 3] * sd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 2, 4]]))

    # column 4
    sd = 0.
    ssd = 0.
    ssd += M[3, 2] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 2] + M[3, 5] * M[4, 2] * M[5, 3]
    ssd -= M[3, 5] * M[4, 3] * M[5, 2] + M[3, 3] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 3]
    sd += M[2, 1] * ssd

    ssd = 0.
    ssd += M[3, 1] * M[4, 3] * M[5, 5] + M[3, 3] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 3]
    ssd -= M[3, 5] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 3]
    sd -= M[2, 2] * ssd

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 5] + M[3, 2] * M[4, 5] * M[5, 1] + M[3, 5] * M[4, 1] * M[5, 2]
    ssd -= M[3, 5] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 5] + M[3, 1] * M[4, 5] * M[5, 2]
    sd += M[2, 3] * ssd

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 3] + M[3, 2] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 2]
    ssd -= M[3, 3] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 3] + M[3, 1] * M[4, 3] * M[5, 2]
    sd -= M[2, 5] * ssd
    d -= M[1, 4] * sd

    # column 5
    sd = 0.
    ssd = 0.
    ssd += M[3, 2] * M[4, 3] * M[5, 4] + M[3, 3] * M[4, 4] * M[5, 2] + M[3, 4] * M[4, 2] * M[5, 3]
    ssd -= M[3, 4] * M[4, 3] * M[5, 2] + M[3, 3] * M[4, 2] * M[5, 4] + M[3, 2] * M[4, 4] * M[5, 3]
    sd += M[2, 1] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [2, 3, 4]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 3] * M[5, 4] + M[3, 3] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 3]
    ssd -= M[3, 4] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 4] + M[3, 1] * M[4, 4] * M[5, 3]
    sd -= M[2, 2] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 3, 4]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 4] + M[3, 2] * M[4, 4] * M[5, 1] + M[3, 4] * M[4, 1] * M[5, 2]
    ssd -= M[3, 4] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 4] + M[3, 1] * M[4, 4] * M[5, 2]
    sd += M[2, 3] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 2, 4]]))

    ssd = 0.
    ssd += M[3, 1] * M[4, 2] * M[5, 3] + M[3, 2] * M[4, 3] * M[5, 1] + M[3, 3] * M[4, 1] * M[5, 2]
    ssd -= M[3, 3] * M[4, 2] * M[5, 1] + M[3, 2] * M[4, 1] * M[5, 3] + M[3, 1] * M[4, 3] * M[5, 2]
    sd -= M[2, 4] * ssd

    println("\tssd ", ssd - det(M[[3, 4, 5], [1, 2, 3]]))

    println("\t", sd, " vs. ", det(M[[2, 3, 4, 5], [1, 2, 3, 4]]))

    d += M[1, 5] * sd

    return d
end
