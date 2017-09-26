using JSON
L = 4

filename = "chain_lattice_L_$(L).json"

n_neighbors = 2
n_sites = L
n_bonds = L
positions = zeros(Int, L, 1)
bonds = -1 * ones(Int, 4, L)
bonds_idx = -1 * ones(Int, 2, L)
bond_types = -1 * ones(Int, L)
vectors = Array{Float64, 2}(L, 2)

s_idx = 1
b_idx = 1

for s in 1:2:L
    positions[b_idx] = s
    bond_types[b_idx] = 1
    bonds[:, b_idx] = [s, 1, mod1(s + 1, L), 1]
    bonds_idx[:, b_idx] = [s, mod1(s + 1, L)]
    vectors[b_idx, :] = [1, 0]
    b_idx += 1

    positions[b_idx] = s + 1
    bond_types[b_idx] = 2
    bonds[:, b_idx] = [mod1(s + 1, L), 1, mod1(s + 2, L), 1]
    bonds_idx[:, b_idx] = [mod1(s + 1, L), mod1(s + 2, L)]
    vectors[b_idx, :] = [1, 0]
    b_idx += 1
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
