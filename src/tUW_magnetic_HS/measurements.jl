import JLD, HDF5

type observables <: observable_type
    sweeps_per_measurement::Int
    batch_size::Int
    mc_observables::Dict{String, observable}
    mc_observable_names::Array{String, 1}
    W_af_fields::Array{Int8, 3}
    U_af_fields::Array{Int8, 3}
    greens_real::Array{Float32, 3}
    greens_imag::Array{Float32, 3}
    observables() = new(1, 1, Dict{String, observable}(), String[], ones(Int8, 1,1,1), ones(Float32, 1,1,1))
end


function initialize_observables(s::stack_type, p::parameter_type, l::lattice)
    obs = observables()
    obs.mc_observable_names = ["Double Occupancy", "Real Spin-Spin", "Imag Spin Spin"]

    for o in obs.mc_observable_names
        obs.mc_observables[o] = observable()
        obs.mc_observables[o].name = o
    end

    ms = min(p.measurements, 128)
    obs.sweeps_per_measurement = 1
    obs.batch_size = ms
    obs.W_af_fields = zeros(size(p.W_af_field)..., ms)
    obs.U_af_fields = zeros(size(p.U_af_field)..., ms)
    obs.greens_real = zeros(l.n_sites, l.n_sites, ms)
    obs.greens_imag = zeros(l.n_sites, l.n_sites, ms)

    return obs
end


function measure(output_file::String, sweep::Int, obs::observable_type, s::stack_type, p::parameter_type, l::lattice)
    if mod(sweep, obs.sweeps_per_measurement) == 0
        if s.current_slice == p.slices / 2 + 1 && s.direction == 1
            measure_double_occupancy(obs, s, p, l)
            measure_long_range_order(obs, s, p, l)
            obs.W_af_fields[:, :, mod1(sweep, obs.batch_size)] = p.W_af_field[:, :]
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
                    JLD.write(f, "simulation/results/W_configurations/$(new_key)", obs.W_af_fields)
                    JLD.write(f, "simulation/results/U_configurations/$(new_key)", obs.U_af_fields)
                    JLD.write(f, "simulation/results/greens_real/$(new_key)", obs.greens_real)
                    JLD.write(f, "simulation/results/greens_imag/$(new_key)", obs.greens_imag)
                    println("Dumping $(new_key) was a success")

                    obs.W_af_fields[:] = 0.
                    obs.U_af_fields[:] = 0.
                    obs.greens_real[:] = 0.
                    obs.greens_imag[:] = 0.
                end
            end
        end
    end
end


function measure_double_occupancy(obs::observable_type, s::stack, p::parameters, l::lattice)
    d_occ = 0.
    for i in 1:l.n_sites      d_occ += (1 - s.greens[i, i])^2     end
    add_value(obs.mc_observables["Double Occupancy"], real(d_occ) / l.n_sites)
end

function measure_long_range_order(obs::observable_type, s::stack, p::parameters, l::lattice)
    # s.greens[o_idx, t_idx] * (1. - s.greens[o_idx, t_idx])
end

function dump_measurements(output_file::String, obs::observable_type, s::stack_type, p::parameter_type, l::lattice)
    println("Dumping measurements")
    for o in values(obs.mc_observables)
        dump_observable(output_file, o)
    end
end
