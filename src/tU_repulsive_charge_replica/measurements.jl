import JLD, HDF5

type observables <: observable_type
    sweeps_per_measurement::Int
    batch_size::Int
    mc_observables::Dict{String, monte_carlo_observable}
    observables() = new(1, 1, Dict{String, monte_carlo_observable}())
end

function initialize_observables(output_file::String, p::parameter_type, l::lattice)
    obs = observables()

    obs.mc_observables["Density"] = monte_carlo_observable{Float64}("Density")
    obs.mc_observables["Double Occupancy"] = monte_carlo_observable{Float64}("Double Occupancy")
    obs.mc_observables["p12"] = monte_carlo_observable{Float64}("p12")
    obs.mc_observables["p21"] = monte_carlo_observable{Float64}("p21")

    load_measurements(output_file, obs)

    ms = min(p.measurements, 128)
    obs.sweeps_per_measurement = 1
    obs.batch_size = ms

    return obs
end


function measure(output_file::String, sweep::Int, obs::observable_type, s::stack_type, s_B::stack_type, p::parameter_type, l::lattice)
    push!(obs.mc_observables["Density"], l.n_sites - real(sum(trace(s.greens))))
    measure_double_occupancy(obs, s, p, l)
    measure_transition_ratio(obs, s, s_B, p, l)
end


function measure_transition_ratio(obs::observable_type, s_A::stack, s_B::stack, p::parameters, l::lattice)
    s_A.Ul, s_A.Dl, s_A.Tl = s_A.u_stack_l[:, :, end], s_A.d_stack_l[:, end], s_A.t_stack_l[:, :, end]
    s_A.Ur, s_A.Dr, s_A.Tr = s_A.u_stack_r[:, :, end], s_A.d_stack_r[:, end], s_A.t_stack_r[:, :, end]
    s_B.Ul, s_B.Dl, s_B.Tl = s_B.u_stack_l[:, :, end], s_B.d_stack_l[:, end], s_B.t_stack_l[:, :, end]
    s_B.Ur, s_B.Dr, s_B.Tr = s_B.u_stack_r[:, :, end], s_B.d_stack_r[:, end], s_B.t_stack_r[:, :, end]

    s_A.u_temp = eye(greens_type, l.n_sites)
    s_A.d_temp = ones(real_type, l.n_sites)
    s_A.t_temp = eye(greens_type, l.n_sites)

    s_B.u_temp = eye(greens_type, l.n_sites)
    s_B.d_temp = ones(real_type, l.n_sites)
    s_B.t_temp = eye(greens_type, l.n_sites)

    calculate_greens_full(s_A, s_B, p, l)
    calculate_greens_full(s_A, p, l)
    calculate_greens_full(s_B, p, l)
    if p.active_configuration == 1
        push!(obs.mc_observables["p12"], min(1., exp(2 * s_A.AB_det - 2 * (s_A.det + s_B.det))))
    else
        push!(obs.mc_observables["p21"], min(1., exp(2 * (s_A.det + s_B.det) - 2 * s_A.AB_det)))
    end
end

function measure_double_occupancy(obs::observable_type, s::stack, p::parameters, l::lattice)
    d_occ = 0.
    for i in 1:l.n_sites      d_occ += (1 - s.greens[i, i])^2     end
    push!(obs.mc_observables["Double Occupancy"], real(d_occ) / l.n_sites)
end


function load_measurements(output_file::String, obs::observable_type)
    println("Loading measurements")
    h = h5open(output_file, "r")
    for o in values(obs.mc_observables)
        read!(h, o)
    end
    close(h)
end


function dump_measurements(output_file::String, obs::observable_type)
    if obs.mc_observables["Density"].n_measurements < 16 return end
    println("Dumping measurements")
    h = h5open(output_file, "r+")

    for o in values(obs.mc_observables)
        write(h, o)
    end
    close(h)
end
