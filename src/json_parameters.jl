import JSON
import HDF5
using Iterators

function load_parameters(filename::String)
    return JSON.parsefile(filename)
end

function write_parameters(prefix::String, params::Dict)
    for (i, param_set) in enumerate(product(values(params)...))
        println(param_set)
        p = Dict{Any, Any}()
        for (k, v) in zip(keys(params), param_set)
            println("Adding $(k) - $(v)")
            p[k] = v
        end
        paramstring = JSON.json(p)
        open(prefix * ".task$(i).in.json", "w") do f
            write(f, paramstring)
        end
    end
end

function parameters2hdf5(params::Dict, filename::String)
  f = HDF5.h5open(filename, "r+")

  for (k, v) in params
    try
      if HDF5.exists(f, "parameters/" * k)
        if read(f["parameters/"*k]) != v
          error(k, " exists but differs from current ")
        end
      else
        f["parameters/" * k] = v
      end
    catch e
    end

  end

  close(f)
end
