import JLD, HDF5

type observables <: observable_type
    sweeps_per_measurement::Int
    batch_size::Int
    mc_observables::Dict{String, monte_carlo_observable}
    U_af_fields::Array{Int8, 3}
    greens_real::Array{Float32, 3}
    greens_imag::Array{Float32, 3}
    observables() = new(1, 1, Dict{String, monte_carlo_observable}(), ones(Int8, 1,1,1), ones(Float32, 1,1,1))
end

function initialize_observables(output_file::String, s::stack_type, p::parameter_type, l::lattice)
    obs = observables()

    obs.mc_observables["Density"] = monte_carlo_observable{Float64}("Density")
    obs.mc_observables["Density Imag"] = monte_carlo_observable{Float64}("Density Imag")
    obs.mc_observables["Double Occupancy"] = monte_carlo_observable{Float64}("Double Occupancy")
    # obs.mc_observables["Real Spin-Spin"] = monte_carlo_observable{Float64}("Real Spin-Spin")
    # obs.mc_observables["Imag Spin-Spin"] = monte_carlo_observable{Float64}("Imag Spin-Spin")
    # obs.mc_observables["Real Pi-Pi"] = monte_carlo_observable{Float64}("Real Pi-Pi")
    # obs.mc_observables["Imag Pi-Pi"] = monte_carlo_observable{Float64}("Imag Pi-Pi")

    load_measurements(output_file, obs)

    ms = min(p.measurements, 128)
    obs.sweeps_per_measurement = 1
    obs.batch_size = ms
    obs.U_af_fields = zeros(size(p.U_af_field)..., ms)
    obs.greens_real = zeros(l.n_sites, l.n_sites, ms)
    obs.greens_imag = zeros(l.n_sites, l.n_sites, ms)

    return obs
end


function measure(output_file::String, sweep::Int, obs::observable_type, s::stack_type, p::parameter_type, l::lattice)
    if mod(sweep, obs.sweeps_per_measurement) == 0
        if s.current_slice == p.slices / 2 + 1 && s.direction == 1
            # if mod(sweep, obs.batch_size) == 0 && sweep > 0
            #     println("Double Occupancy\t", mean(obs.mc_observables["Double Occupancy"].timeseries))
            #     println("Real Spin Spin\t", mean(obs.mc_observables["Real Spin-Spin"].timeseries))
            #     println("Real Pi-Pi\t", mean(obs.mc_observables["Real Pi-Pi"].timeseries))
            # end
            push!(obs.mc_observables["Density"], real(l.n_sites - sum(trace(s.greens))))
            push!(obs.mc_observables["Density Imag"], imag(l.n_sites - sum(trace(s.greens))))
            measure_double_occupancy(obs, s, p, l)
            # measure_pi_pi(obs, s, p, l)
            # measure_long_range_order(obs, s, p, l)
            obs.U_af_fields[:, :, mod1(sweep, obs.batch_size)] = p.U_af_field[:, :]
            obs.greens_real[:, :, mod1(sweep, obs.batch_size)] = real(s.greens[:, :])
            obs.greens_imag[:, :, mod1(sweep, obs.batch_size)] = imag(s.greens[:, :])

            if mod(sweep, obs.batch_size) == 0 && sweep > 0
                JLD.jldopen(output_file, "r+", compress=true) do f
                    new_key = string(Int(sweep / obs.batch_size))

                    if HDF5.exists(f, "simulation/results/configurations/$(new_key)")
                        ks = sort([parse(Int64, string(x)) for x in HDF5.names(f["simulation/results/configurations/"])])
                        new_key = string(ks[end] + 1)
                    end
                    JLD.write(f, "simulation/results/U_configurations/$(new_key)", obs.U_af_fields)
                    JLD.write(f, "simulation/results/greens_real/$(new_key)", obs.greens_real)
                    JLD.write(f, "simulation/results/greens_imag/$(new_key)", obs.greens_imag)
                    println("Dumping $(new_key) was a success")

                    obs.U_af_fields[:] = 0.
                    obs.greens_real[:] = 0.
                    obs.greens_imag[:] = 0.
                end
                dump_measurements(output_file, obs, s, p, l)

            end
        end
    end
end


function measure_double_occupancy(obs::observable_type, s::stack, p::parameters, l::lattice)
    d_occ = 0.
    for i in 1:l.n_sites      d_occ += (1 - s.greens[i, i])^2     end
    push!(obs.mc_observables["Double Occupancy"], real(d_occ) / l.n_sites)
end

function measure_long_range_order(obs::observable_type, s::stack, p::parameters, l::lattice)
    correlation = 0. + 0im
    r = 1
    c = 0
    # for r::Int in 1:l.L
    #     for c::Int in 0:l.L-1
            o = rc_to_idx(r, c, l.L)
            t = rc_to_idx(mod1(Int(r + l.L /2 ), l.L), c, l.L)
            kr = o == t ? 1. : 0.
            correlation += 1 /3 * s.greens[o, t] * (kr - s.greens[o, t])
            t = rc_to_idx(mod1(Int(r - l.L /2), l.L), Int(c), l.L)
            correlation += 1 /3 * s.greens[o, t] * (kr - s.greens[o, t])
            t = rc_to_idx(r, mod(Int(c + l.L/2), l.L), l.L)
            correlation += 1 /3 * s.greens[o, t] * (kr - s.greens[o, t])
            t = rc_to_idx(r, mod(Int(c - l.L/2), l.L), l.L)
            correlation += 1 /3 * s.greens[o, t] * (kr - s.greens[o, t])
    #     end
    # end
    # add_value(obs.mc_observables["Real Spin-Spin"], real(correlation))
    # add_value(obs.mc_observables["Imag Spin-Spin"], imag(correlation))
    # s.greens[o_idx, t_idx] * (1. - s.greens[o_idx, t_idx])
end

function measure_pi_pi(obs::observable_type, s::stack, p::parameters, l::lattice)
    origin = [0, 0]
    o_idx = 0
    target = [0, 0]
    t_idx = 0
    q = [pi, pi]
    s_factor = complex(0., 0.)
    L = l.L
    try
        for r1 in 1:L, c1 in 1:L
            for r2 in 1:L, c2 in 1:L
                o_idx = (r1  + (c1 - 1) * L)
                origin = [r1, c1]
                t_idx = (r2  + (c2 - 1) * L)
                target = [r2, c2]
                kr = o_idx == t_idx ? 1. : 0.
                s_factor += exp(complex(0., vecdot(q, origin - target))) * s.greens[o_idx, t_idx] * (kr - s.greens[o_idx, t_idx])
            end
        end
    catch e end
    # add_value(obs.mc_observables["Real Pi-Pi"], real(s_factor))
    # add_value(obs.mc_observables["Imag Pi-Pi"], imag(s_factor))
end

function load_measurements(output_file::String, obs::observable_type)
    println("Loading measurements")
    h = h5open(output_file, "r")
    for o in values(obs.mc_observables)
        read!(h, o)
    end
    close(h)
end


function dump_measurements(output_file::String, obs::observable_type, s::stack_type, p::parameter_type, l::lattice)
    if obs.mc_observables["Density"].n_measurements < 16 return end
    println("Dumping measurements")
    h = h5open(output_file, "r+")
    for o in values(obs.mc_observables)
        write(h, o)
    end
    close(h)
end
