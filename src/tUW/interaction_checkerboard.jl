function initialize_interaction_groups(s::stack_type, p::parameter_type, l::lattice)
    site_group = zeros(Int, l.n_sites)
    n_groups = 0
    n_max_sites_per_group = 0

    while minimum(site_group) == 0 && n_groups < 50
        n_groups += 1

        site_is_neighbor = zeros(Int, l.n_sites)
        n_sites_per_group = 0

        for i in 1:l.n_sites
            if site_group[i] != 0 || site_is_neighbor[i] != 0 continue end
            row, col = mod1(i, l.L), Int(ceil(i / l.L)) - 1

            if row + col * l.L != i
                println("Error assigning row, col from index")
                println("$(row), $(col) != $(i)")
            end

            add_site_to_group = true
            neighbors = [mod1(row + 1, l.L) + col * l.L, mod1(row - 1, l.L) + col * l.L, row + mod(col + 1, l.L) * l.L, row + mod(col + l.L - 1, l.L) * l.L]
            for j in neighbors
                if site_is_neighbor[j] == 1
                    add_site_to_group = false
                    break
                end
            end

            if add_site_to_group == true
                site_is_neighbor[i] = 1
                site_is_neighbor[neighbors] = 1
                site_group[i] = n_groups
                n_sites_per_group += 1
            end
        end
        n_max_sites_per_group = max(n_sites_per_group, n_max_sites_per_group)
    end

    s.site_group_assignment = zeros(Int, n_groups, n_max_sites_per_group)
    for r in 1:n_groups
        idx = 1
        for c in 1:l.n_sites
            if site_group[c] == r
                s.site_group_assignment[r, idx] = c
                idx += 1
            end
        end
    end
end
