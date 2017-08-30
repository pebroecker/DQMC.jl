using Iterators

code_dir = ENV["CODE_DIR"]
lattice_dir = ENV["LATTICE_DIR"]
include("$(code_dir)/json_parameters.jl")

for L in [4]
    W = L
    beta = 10
    dt = 0.1

    dir = "L_$(L)_W_$(W)_gs"
    if !isdir(dir) mkdir(dir) end
    cd(dir)
    prefix = "square_L_$(L)_W_$(W)_beta_$(beta)_dt_$(dt)"
    if !(isdir(prefix)) mkdir(prefix) end
    cd(prefix)

    p = Dict{Any, Any}("LATTICE_FILE"=>["$(lattice_dir)/square_lattice_L_$(L)_W_$(W).json"], "SLICES"=>[Int(beta / dt)], "DELTA_TAU"=>[dt], "SAFE_MULT"=>[10], "U"=>[8.0], "SEED"=>[13])
    p["LATTICE"] = ["square"]
    p["MODEL"] = ["tU_attractive_charge"]
    p["STACK_HANDLING"] = ["ground_state"]
    p["THERMALIZATION"] = 256
    p["MEASUREMENTS"] = 4096
    p["PARTICLES"] = 
    p["HOPPINGS"] =  ["1.0,1.0,1.0"]
    write_parameters(prefix, p)


    job_cheops = """
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=24:00:00
#SBATCH --account=UniKoeln

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source ~/.bashrc

cd /projects/ag-trebst/pboertz/julia_dqmc/honeycomb/$(dir)/$(prefix)/
julia /projects/ag-trebst/pboertz/codes/julia_dqmc/src/dqmc.jl $(prefix) \$\{SLURM_ARRAY_TASK_ID\}
    """
    f = open("$(prefix).sh", "w")
    write(f, job_cheops)
    close(f)

    cd("../..")
end
