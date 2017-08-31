using HDF5

in_file = ARGS[1]
L = parse(Int, ARGS[2])
out_file = "qlt_$(in_file)"

h = h5open(in_file, "r")
# 64x2x25x2x3000
data = zeros(L * L, 1, 25, 1, 32 * 128)

for i in 1:32
    greens = read(h, "simulation/results/greens_real/1")# + 1im * read(h, "simulation/results/greens_imag/1")
    neighs = [-L, -1, 0, 1, L]

    for site in 1:L * L
        ns = mod1.(site + neighs, L * L)
        idx = 1
        for (k, m) in enumerate(ns)
            for (l, n) in enumerate(ns)
                for sample in 1:128
                    data[site, 1, 5 * (k - 1) + l, 1, idx] = greens[m, n, sample]
                    idx += 1
                end
            end
        end
    end
end

out_h = h5open(out_file, "w")
write(out_h, "correlations", data)
HDF5.o_copy(h, "simulation/results/Density", out_h, "Density")#, HDF5.H5P_OBJECT_COPY, HDF5.H5P_LINK_CREATE)
close(h)
