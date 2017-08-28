type lattice
  n_sites::Int
  n_neighbors::Int
  n_bonds::Int
  L::Int
  positions::Array{Int, 2}
  bonds::Array{Int, 2}
  bonds_idx::Array{Int, 2}
  site_bonds::Array{Int, 2}
  vectors::Array{Float64, 2}
  lattice() = new()
end


function get_lattice(params::Dict)
    l = lattice()
    lat = JSON.parsefile(params["LATTICE_FILE"])
    l.n_sites = Int(lat["n_sites"])
    l.n_neighbors = Int(lat["n_neighbors"])
    l.n_bonds = Int(lat["n_bonds"])
    l.positions = hcat(lat["positions"]...)
    l.bonds = hcat(lat["bonds"]...)
    l.bonds_idx = hcat(lat["bonds_idx"]...)
    l.vectors = hcat(lat["vectors"]...)
    return l
end
