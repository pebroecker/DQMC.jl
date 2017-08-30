type parameters <: parameter_type
    stack_handling::String
    particles::Int
    hoppings::Array{Float64, 1}
    U::Float64
    lambda::Float64
    delta_tau::Float64
    theta::Float64
    mu::Float64
    slices::Int
    safe_mult::Int
    U_af_field::Array{Float64, 2}
    curr_sweep::Int
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
    p.stack_handling = params["STACK_HANDLING"]
    p.curr_sweep = 1
    p.thermalization = Int(params["THERMALIZATION"])
    p.measurements = Int(params["MEASUREMENTS"])
    p.slices = Int(params["SLICES"])
    p.delta_tau = Float64(params["DELTA_TAU"])
    p.safe_mult = Int(params["SAFE_MULT"])
    p.theta = p.slices * p.delta_tau
    p.U = Float64(params["U"])
    p.lattice_name = params["LATTICE"]
    p.seed = Int(params["SEED"])
    p.r = srand(p.seed)
    # parameter for Hubbard-Stratonovich transformation | SU(2) invariant version
    p.lambda = acosh(exp(p.U * p.delta_tau / 2))
    p.hoppings = [parse(Float64, h) for h in split(params["HOPPINGS"], ",")]
    # in case we want to move away from half filling
    p.mu = "MU" in keys(params) ? Float64(params["MU"]) : 0.0

    println("Working with U = $(p.U) and mu = $(p.mu)")
    return p
end


function initialize_lattice_parameters(p::parameters, l::lattice)
    p.particles = "PARTICLES" in keys(params) ? params["PARTICLES"] : l.n_sites
    p.U_af_field = rand([-1, 1], l.n_sites, p.slices)
end


function load_state(output_file::String, p::parameters)
    last_state = jldopen(output_file, "r")
    if !exists(last_state, "simulation/state/curr_sweep") return end

    p.curr_sweep = read(last_state, "simulation/state/curr_sweep")
    p.U_af_field = read(last_state, "simulation/state/U_configuration")
    p.r = read(last_state, "simulation/state/rng")
    println("Last dump was after $(p.curr_sweep) sweeps")
    close(last_state)
end

function save_state(output_file::String, p::parameters)
    h = jldopen(output_file, "r+")
    if(exists(h, "simulation/state/")) o_delete(h, "simulation/state") end
    write(h, "simulation/state/curr_sweep", p.curr_sweep)
    write(h, "simulation/state/U_configuration", p.U_af_field)
    write(h, "simulation/state/rng", p.r)
    close(h)
end
