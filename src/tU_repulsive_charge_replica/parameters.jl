
type parameters <: parameter_type
    stack_handling::String

    particles::Int
    hoppings::Array{Float64, 1}

    U::Float64
    t::Float64
    lambda::Complex{Float64}
    delta_tau::Float64
    theta::Float64
    mu::Float64
    slices::Int
    safe_mult::Int
    U_af_field_A::Array{Float64, 2}
    U_af_field_B::Array{Float64, 2}
    U_af_field_A_1::Array{Float64, 2}
    U_af_field_A_2::Array{Float64, 2}
    U_af_field_B_1::Array{Float64, 2}
    U_af_field_B_2::Array{Float64, 2}
    thermalization::Int
    measurements::Int
    lattice_name::String
    seed::Int
    n_A::Int
    n_B::Int
    N::Int
    curr_sweep::Int
    replica::Int
    active_configuration::Int
    parameters() = new()
end

function get_parameters(params::Dict)
    # Assign parameters based on XML file
    p = parameters()
    p.stack_handling = params["STACK_HANDLING"]
    p.thermalization = Int(params["THERMALIZATION"])
    p.measurements = Int(params["MEASUREMENTS"])
    p.slices = Int(params["SLICES"])
    p.delta_tau = Float64(params["DELTA_TAU"])
    p.safe_mult = Int(params["SAFE_MULT"])
    p.theta = p.slices * p.delta_tau
    p.U = Float64(params["U"])
    p.t = Float64(params["t"])
    p.lattice_name = params["LATTICE"]
    p.seed = Int(params["SEED"])
    # parameter for Hubbard-Stratonovich transformation | SU(2) invariant version
    p.lambda = acosh(exp(-p.U * p.delta_tau / 2 + 0im))
    p.hoppings = [parse(Float64, h) for h in split(params["HOPPINGS"], ",")]
    p.curr_sweep = 1
    p.n_A = Int(params["CUT_STEP"])

    # in case we want to move away from half filling
    if "MU" in keys(params)
        p.mu = Float64(params["MU"])
    else
        p.mu = 0.0
    end

    if p.stack_handling == "ground_state"
        p.particles = Int(params["PARTICLES"])
    end

    println("Working with U = $(p.U) and mu = $(p.mu). Lambda is $(p.lambda)")

    if p.stack_handling == "ground_state"
        println("Doing a ground state simulation with $(p.particles) particles")
    end

    return p
end

function initialize_lattice_parameters(p::parameters, l::lattice)

    p.n_B = l.n_sites - p.n_A
    p.N = p.n_A + 2 * p.n_B

    println("$(p.n_A)\t$(p.n_B)\t$(p.N)\t$(l.n_sites)")

    if p.stack_handling != "ground_state"
        p.particles = l.n_sites
    end

    p.U_af_field_A = rand([-1, 1], l.n_sites, p.slices)
    p.U_af_field_B = rand([-1, 1], l.n_sites, p.slices)
    p.U_af_field_A_1 = rand([-1, 1], l.n_sites, p.slices)
    p.U_af_field_A_2 = rand([-1, 1], l.n_sites, p.slices)
    p.U_af_field_B_1 = rand([-1, 1], l.n_sites, p.slices)
    p.U_af_field_B_2 = rand([-1, 1], l.n_sites, p.slices)
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
