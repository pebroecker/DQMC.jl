
type parameters <: parameter_type
  particles::Int
  U::Float64
  t::Float64
  phi::Float64
  lambda::Complex{Float64}
  delta_tau::Float64
  theta::Float64
  mu::Float64
  slices::Int
  safe_mult::Int
    W_af_field::Array{Float64, 2}
      U_af_field::Array{Float64, 2}
  thermalization::Int
  measurements::Int
  lattice_name::String
  seed::Int
  parameters() = new()
end

function get_parameters(params::Dict)
    # Assign parameters based on XML file
    p = parameters()
    p.thermalization = Int(params["THERMALIZATION"])
    p.measurements = Int(params["MEASUREMENTS"])
    p.slices = Int(params["SLICES"])
    p.delta_tau = Float64(params["DELTA_TAU"])
    p.safe_mult = Int(params["SAFE_MULT"])
    p.theta = p.slices * p.delta_tau
    p.U = Float64(params["U"])
    p.t = Float64(params["t"])
    p.phi = Float64(params["phi"])
    p.lattice_name = params["LATTICE"]
    p.seed = Int(params["SEED"])
    # parameter for Hubbard-Stratonovich transformation | SU(2) invariant version
    p.lambda = acosh(exp(-p.U * p.delta_tau / 2 + 0im))

    # in case we want to move away from half filling
    if "MU" in keys(params)
        p.mu = Float64(params["MU"])
    else
        p.mu = 0.0
    end
    println("Working with U = $(p.U) and mu = $(p.mu). Lambda is $(p.lambda)")
    return p
end

function initialize_lattice_parameters(p::parameters, l::lattice)
    l.L = Int(sqrt(l.n_sites))
    if p.stack_handling == "ground_state"
        p.particles = Int(l.n_sites / 2)
    else
        p.particles = l.n_sites
    end
    p.U_af_field = rand([-1, 1], l.n_sites, p.slices)
end
