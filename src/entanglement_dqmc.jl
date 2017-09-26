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

params = load_parameters(input_file)
model_name = String(params["MODEL"])
lattice_name = String(params["LATTICE"])
stack_handling = String(params["STACK_HANDLING"])

include("$(model_name)/types.jl")
include("la.jl")
include("lattice.jl")
include("replica_stack.jl")
include("$(model_name)/parameters.jl")
include("$(model_name)/stack.jl")
include("$(model_name)/updates.jl")
include("$(model_name)/measurements.jl")
include("entanglement_replica.jl")

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


        s_A = stack()
        s_B = stack()
        initialize_stack(s_A, p, l)
        initialize_model_stack(s_A, p, l)
        s_A.replica = 1
        initialize_stack(s_B, p, l)
        initialize_model_stack(s_B, p, l)
        s_B.replica = 2
        test_stack(s_A, p, l)
        test_stack(s_B, p, l)

        build_stack(s_A, p, l)
        build_stack(s_B, p, l)

        swept_on_slice_A = zeros(p.slices)
        swept_on_slice_B = zeros(p.slices)

        # for i in 1:(3 * p.slices + 50)
        #     # println("Calling update i = $(i)",)
        #     update(s_A, p, l)
        #     swept_on_slice[s_A.current_slice] += 1
        # end
        # println(minimum(swept_on_slice), " - ", maximum(swept_on_slice))
        # println("Propagation done")
        # obs = initialize_observables(output_file, s, p, l)
        obs = initialize_observables(output_file, p, l)
        sweeps = 2 * p.thermalization + p.measurements
        msrmnt(x) = x - p.thermalization

        p.active_configuration = 2

        try
            for sweep in 1:sweeps
                if sweep % 256 == 0
                    println("Switching from configuration $(p.active_configuration)")
                    if (minimum(swept_on_slice_A) != maximum(swept_on_slice_A)) || (maximum(swept_on_slice_A) != maximum(swept_on_slice_B))
                        println("Swept incorrectly")
                        println(minimum(swept_on_slice_A), maximum(swept_on_slice_A))
                        println(minimum(swept_on_slice_B), maximum(swept_on_slice_B))
                    end

                    if p.active_configuration == 1
                        p.U_af_field_A_1[:] = p.U_af_field_A[:]
                        p.U_af_field_B_1[:] = p.U_af_field_B[:]
                        p.U_af_field_A[:] = p.U_af_field_A_2[:]
                        p.U_af_field_B[:] = p.U_af_field_B_2[:]

                    else
                        p.U_af_field_A_2[:] = p.U_af_field_A[:]
                        p.U_af_field_B_2[:] = p.U_af_field_B[:]
                        p.U_af_field_A[:] = p.U_af_field_A_1[:]
                        p.U_af_field_B[:] = p.U_af_field_B_1[:]
                    end
                    p.active_configuration = mod1(p.active_configuration + 1, 2)
                    build_stack(s_A, p, l)
                    build_stack(s_B, p, l)
                    swept_on_slice_A[:] = 0
                    swept_on_slice_B[:] = 0
                end

                for slice in 1:2 * p.slices
                    if p.active_configuration == 1
                        if (sweep % 256 <= 128)
                            update(s_A, p, l)
                            swept_on_slice_A[s_A.current_slice] += 1
                        else
                            update(s_B, p, l)
                            swept_on_slice_B[s_B.current_slice] += 1
                        end
                    else
                        if (sweep % 256 <= 128)
                            update(s_A, s_B, p, l)
                            swept_on_slice_A[s_A.current_slice] += 1
                        else update(s_B, s_A, p, l)
                            swept_on_slice_B[s_B.current_slice] += 1
                        end
                    end
                end

                if msrmnt(p.curr_sweep) > 0
                    measure(output_file, msrmnt(p.curr_sweep), obs, s_A, s_B, p, l)
                end

                p.curr_sweep += 1
                if p.curr_sweep % 256 == 0
                    println(p.curr_sweep)
                end

                if msrmnt(p.curr_sweep) == p.measurements
                    println("Finished $(p.measurements) measurements")
                    dump_measurements(output_file, obs)
                end
            end
        catch e
            if isa(e, InterruptException)
                save_state(output_file, p)
                dump_measurements(output_file, obs)
            else
                display(catch_stacktrace())
                println(e)
            end
        end
    end

    main(ARGS)
end
