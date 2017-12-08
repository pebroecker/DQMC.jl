using HDF5

in_file = ARGS[1]
L = parse(Int, ARGS[2])
out_file = "qlt_$(in_file)"

h = h5open(in_file, "r")

n_samples = 32

data = zeros(Complex{Float32}, L * L, 1, 25, 1, n_samples * 128)


for i in 1:n_samples
    greens = read(h, "simulation/results/greens_real/$(i)") + 1im * read(h, "simulation/results/greens_imag/$(i)")

    for r in 1:L
        for c in 1:L
            idx = r + (c - 1) * L
            n1 = mod1(r + 1, L)  + (c - 1) * L
            n2 = mod1(r - 1 + L, L)  + (c - 1) * L
            n3 = r + mod(c, L) * L
            n4 = r + mod(c - 2 + L, L) * L
            
            ns = [idx, n1, n2, n3, n4]

            println(ns)

            for (k, m) in enumerate(ns)
                for (l, n) in enumerate(ns)
                    for sample in 1:128
                        data[site, 1, 5 * (k - 1) + l, 1, (i - 1) * 128 + sample] = greens[m, n, sample]
                    end
                end
            end
        end
    end
end

out_h = h5open(out_file, "w")
write(out_h, "correlations_real", real(data))
write(out_h, "correlations_imag", imag(data))
HDF5.o_copy(h, "simulation/results/Density", out_h, "Density")#, HDF5.H5P_OBJECT_COPY, HDF5.H5P_LINK_CREATE)
close(h)
