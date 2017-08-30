using JSON
L = 8

filename = "chain_lattice_L_$(L).json"

n_neighbors = 2
n_sites = L
n_bonds = L
positions = zeros(Int, L, 1)
bonds = -1 * ones(Int, 2, L)
bonds_idx = -1 * ones(Int, 2, L)
bond_types = -1 * ones(Int, L)
vectors = Array{Float64, 2}(L, 2)

s_idx = 1
b_idx = 1

for s in 1:L
    positions[s] = s
    bond_types[s] = 1
    bonds[:, b_idx] = [r, c, r, mod1(c + 1, L)]
    bonds_idx[:, b_idx] = [(r - 1) * L + c, (r - 1) * L + mod1(c + 1, L)]
    vectors[b_idx, :] = [0, 1]
    b_idx += 1
end

lattice = Dict{String, Any}()
lattice["n_sites"] = n_sites
lattice["n_neighbors"] = n_neighbors
lattice["n_bonds"] = n_bonds
lattice["positions"] = positions
lattice["bonds"] = bonds
lattice["bonds_idx"] = bonds_idx
lattice["vectors"] = vectors

latticestring = json(lattice)
open(filename, "w") do f
    write(f, latticestring)
end
