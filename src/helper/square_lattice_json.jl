using JSON
L = 8

filename = "square_lattice_L_$(L)_W_$(L).json"

n_neighbors = 4
n_sites = L * L
n_bonds = 2 * L * L
positions = zeros(Int, L^2, 2)
bonds = -1 * ones(Int, 4, 2 * (L * L))
bonds_idx = -1 * ones(Int, 2, 2 * (L * L))
bond_types = -1 * ones(Int, 2 * (L * L))
vectors = Array{Float64, 2}(2 * L^2, 2)

s_idx = 1
b_idx = 1

for r in 1:L
    for c in 1:L
        positions[s_idx, :] = [r, c]
        s_idx += 1

        bond_types[b_idx] = 1
        bonds[:, b_idx] = [r, c, r, mod1(c + 1, L)]
        bonds_idx[:, b_idx] = [(r - 1) * L + c, (r - 1) * L + mod1(c + 1, L)]
        vectors[b_idx, :] = [0, 1]
        b_idx += 1

        bond_types[b_idx] = 2
        bonds[:, b_idx] = [r, c, mod1(r + 1, L), c]
        bonds_idx[:, b_idx] = [(r - 1) * L + c, (mod1(r + 1, L) - 1) * L + c]
        vectors[b_idx, :] = [1, 0]
        b_idx += 1
    end
end

lattice = Dict{String, Any}()
lattice["n_sites"] = n_sites
lattice["n_neighbors"] = n_neighbors
lattice["n_bonds"] = n_bonds
lattice["positions"] = positions
lattice["bonds"] = bonds
lattice["bonds_idx"] = bonds_idx
lattice["bond_types"] = bond_types
lattice["vectors"] = vectors

latticestring = json(lattice)
open(filename, "w") do f
    write(f, latticestring)
end
