type parameters <: parameter_type
  particles::Int
  U::Float64
  lambda::Float64
  delta_tau::Float64
  theta::Float64
  mu::Float64
  slices::Int
  safe_mult::Int
  af_field::Array{Float64, 2}
  thermalization::Int
  measurements::Int
  lattice_name::String
  seed::Int
  r::MersenneTwister

  parameters() = new()
end

function get_parameters(params::Dict)
    # Assign parameters based on XML file
    p = parameters()
    p.curr_sweep = 1
    p.thermalization = Int(params["THERMALIZATION"][1])
    p.measurements = Int(params["MEASUREMENTS"][1])
    p.slices = Int(params["SLICES"][1])
    p.delta_tau = Float64(params["DELTA_TAU"][1])
    p.safe_mult = Int(params["SAFE_MULT"][1])
    p.theta = p.slices * p.delta_tau
    p.U = Float64(params["U"][1])
    p.lattice_name = params["LATTICE"][1]
    p.seed = Int(params["SEED"][1])
    p.r = srand(p.seed)
    # parameter for Hubbard-Stratonovich transformation | SU(2) invariant version
    p.lambda = acosh(exp(p.U * p.delta_tau / 2))

    # in case we want to move away from half filling
    if "MU" in keys(params)
        p.mu = Float64(params["MU"][1])
    else
        p.mu = 0.0
    end
    println("Working with U = $(p.U) and mu = $(p.mu)")
    return p
end


function initialize_lattice_parameters(p::parameters, l::lattice)
    p.particles = l.n_sites
    p.af_field = rand([-1, 1], l.n_sites, p.slices)
end


function load_state(output_file::String, p::parameters)
    last_state = jldopen(output_file, "r")
    p.curr_sweep = read(last_state, "simulation/state/curr_sweep")
    p.af_field = read(last_state, "simulation/state/U_configuration")
    p.r = read(last_state, "simulation/state/rng")
    println("Last dump was after $(curr_sweep) sweeps")
    close(last_state)
end

function save_state(output_file::String, p::parameters)
    h = jldopen(output_file, "r+")
    if(exists(h, "simulation/state/")) o_delete(h, "simulation/state") end
    write(h, "simulation/state/curr_sweep", p.curr_sweep)
    write(h, "simulation/state/U_configuration", p.af_field)
    write(h, "simulation/state/rng", p.r)
    close(h)
end
