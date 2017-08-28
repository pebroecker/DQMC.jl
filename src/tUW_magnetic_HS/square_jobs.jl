using Iterators

code_dir = ENV["CODE_DIR"]
lattice_dir = ENV["LATTICE_DIR"]
include("$(code_dir)/json_parameters.jl")

for L in [4, 6, 8]
    W = L
    beta = 20
    dt = 0.05
    U = 4.0
    phi = 0.

    dir = "L_$(L)_W_$(W)"
    if !isdir(dir) mkdir(dir) end
    cd(dir)
    prefix = "square_L_$(L)_W_$(W)_beta_$(beta)_dt_$(dt)_U_$(U)"
    if !(isdir(prefix)) mkdir(prefix) end
    cd(prefix)

    p = Dict{Any, Any}("LATTICE_FILE"=>["$(lattice_dir)/square_lattice_L_$(L)_W_$(W).json"], "SLICES"=>[Int(beta / dt)])
    p["DELTA_TAU"] = [dt]
    p["SAFE_MULT"] = [10]
    p["U"] =  [4]
    p["W"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    p["t"] = 1.
    p["phi"] = phi * 2 * pi
    p["HOPPINGS"] = ["1.0,1.0,1.0"]
    p["MODEL"] = ["tUW_magnetic_HS"]
    p["LATTICE"] = ["square"]
    p["STACK_HANDLING"] = ["blank"]
    p["SEED"] = [13]
    p["THERMALIZATION"] = 256
    p["MEASUREMENTS"] = 4096
    write_parameters(prefix, p)


    job_cheops = """
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=$(Int(L/2))gb
#SBATCH --time=$(Int(L^2)):00:00
#SBATCH --account=UniKoeln

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source ~/.bashrc

cd /projects/ag-trebst/pboertz/julia_dqmc/tUW/$(dir)/$(prefix)/
julia /projects/ag-trebst/pboertz/codes/julia_dqmc/src/dqmc.jl $(prefix) \$\{SLURM_ARRAY_TASK_ID\}
    """
    f = open("$(prefix).sh", "w")
    write(f, job_cheops)
    close(f)

    cd("../..")
end
