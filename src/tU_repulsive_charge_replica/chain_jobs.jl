using Iterators

code_dir = ENV["CODE_DIR"]
lattice_dir = ENV["LATTICE_DIR"]
include("$(code_dir)/json_parameters.jl")

for L in [8]
    beta = 20
    dt = 0.1
    U = 4.0

    dir = "L_$(L)"
    if !isdir(dir) mkdir(dir) end
    cd(dir)
    prefix = "chain_L_$(L)_beta_$(beta)_dt_$(dt)_U_$(U)"
    if !(isdir(prefix)) mkdir(prefix) end
    cd(prefix)

    p = Dict{Any, Any}("LATTICE_FILE"=>["$(lattice_dir)/chain_lattice_L_$(L).json"], "SLICES"=>[Int(beta / dt)])
    p["DELTA_TAU"] = [dt]
    p["SAFE_MULT"] = [10]
    p["U"] =  [4]
    p["t"] = 1.
    p["HOPPINGS"] = ["1.0,1.0"]
    p["MODEL"] = ["tU_repulsive_charge_replica"]
    p["LATTICE"] = ["chain"]
    p["STACK_HANDLING"] = ["ground_state"]
    p["SEED"] = [13]
    p["THERMALIZATION"] = 256
    p["MEASUREMENTS"] = 4096
    p["CUT_STEP"] = collect(1:L)
    p["PARTICLES"] = [Int(L / 2)]
    write_parameters(prefix, p)

    cd("../..")
end
