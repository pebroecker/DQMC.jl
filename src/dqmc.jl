ccall(:jl_exit_on_sigint, Void, (Cint,), 0)
using JLD, HDF5
using MonteCarloObservable

include("types.jl")
include("json_parameters.jl")

Base.show(io::IO, f::Float64) = @printf io "%.3e" round(f, 3) #print(io, string(round(f,2)))

prefix = convert(String, ARGS[1]) # which prefix should be used for input files
idx = parse(Int, ARGS[2]) # index for input file
input_file = prefix * ".task" * string(idx) * ".in.json"
output_file = prefix * ".task" * string(idx) * ".out.h5"

println("Processing input file $(input_file)")
params = load_parameters(input_file)
model_name = String(params["MODEL"])
lattice_name = String(params["LATTICE"])
stack_handling = String(params["STACK_HANDLING"])

include("$(model_name)/types.jl")
include("la.jl")
include("lattice.jl")
include("stack.jl")
include("$(stack_handling).jl")
include("$(model_name)/parameters.jl")
include("$(model_name)/stack.jl")
include("$(model_name)/updates.jl")
include("$(model_name)/measurements.jl")

#  disable bounds check for array access
@inbounds begin
    function main(ARGS::Array{String, 1})
        if !isfile(output_file)
            close(JLD.jldopen(output_file, "w", compress=true))
        end

        try     parameters2hdf5(params, output_file) # json_parameters
        catch e     println(e)      end

        p = get_parameters(params)  # model/parameters.jl
        l = get_lattice(params)     # lattice.jl
        initialize_lattice_parameters(p, l)
        load_state(output_file, p)


        s = stack()
        initialize_stack(s, p, l)
        initialize_model_stack(s, p, l)
        test_stack(s, p, l)
        build_stack(s, p, l)
        propagate(s, p, l)

        obs = initialize_observables(output_file, s, p, l)

        sweeps = p.thermalization + p.measurements
        msrmnt(x) = x - p.thermalization

        try
            for _ in 1:sweeps
                for u in 1:2 * p.slices
                    update(s, p, l)

                    if msrmnt(p.curr_sweep) > 0
                        measure(output_file, msrmnt(p.curr_sweep), obs, s, p, l)
                    end
                end

                if msrmnt(p.curr_sweep) > p.measurements
                    println("Finished $(p.measurements) measurements")
                    dump_measurements(output_file, obs, s, p, l)
                    break
                end

                p.curr_sweep += 1
                if p.curr_sweep % 256 == 0
                    println(p.curr_sweep)
                end
            end
        catch e
            if isa(e, InterruptException)
                save_state(output_file, p)
                dump_measurements(output_file, obs, s, p, l)
            else
                display(catch_stacktrace())
                println(e)
            end
        end
    end

    main(ARGS)
end
