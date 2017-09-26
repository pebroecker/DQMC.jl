################################################################################
# Turning small to large matrices
################################################################################

function col_to_invertible(M::Array{greens_type, 2}, p::parameters, l::lattice)
    out = rand(greens_type, l.n_sites, l.n_sites)
    out[:, 1:p.particles] = M

    for c in p.particles + 1:l.n_sites
        for c_o in 1:c - 1
            projection = sum(out[:, c] .* conj(out[:, c_o]))
            out[:, c] -= projection * out[:, c_o]
        end
        out[:, c] /= norm(out[:, c]) # sum(out[:, c] .* conj(out[:, c]))
    end

    return out
end

function enlarge(A::Array{greens_type, 2}, B::Array{greens_type, 2}, replica::Int, p::parameters, l::lattice)
    if replica == 1
        B[:] = 0.
        B[1:l.n_sites, 1:l.n_sites] = A[1:l.n_sites, 1:l.n_sites]
        # println("$(l.n_sites) physical sites with $(p.N) total sites and $(p.n_B) - $(size(B))")
        B[l.n_sites + 1:p.N, l.n_sites + 1:p.N] = eye(p.n_B)
    else
        B[:] = 0
        B[1:p.n_A, 1:p.n_A] = A[1 : p.n_A, 1:p.n_A]
        B[p.n_A+1 : l.n_sites, p.n_A + 1:l.n_sites] = eye(p.n_B)
        B[l.n_sites+1 : p.N, 1 : p.n_A] = A[p.n_A+1 : l.n_sites, 1 : p.n_A]
        B[1:p.n_A, l.n_sites+1 : p.N] = A[1 : p.n_A, p.n_A+1 : l.n_sites]
        B[l.n_sites+1 : p.N, l.n_sites+1 : p.N] = A[p.n_A+1 : l.n_sites, p.n_A+1 : l.n_sites]
    end
end

function enlarge(A::Array{real_type, 1}, B::Array{real_type, 1}, replica::Int, p::parameters, l::lattice)
    if replica == 1
        B[:] = 1.
        B[1:l.n_sites] = A[:]
    else
        B[:] = 1.
        B[1:p.n_A] = A[1:p.n_A]
        B[l.n_sites+1 : p.N] = A[p.n_A+1 : l.n_sites]

    end
end

function enlarge_thin(A::Array{greens_type, 2}, B::Array{greens_type, 2}, replica::Int, p::parameters, l::lattice)
    if replica == 1
        B[:] = 0.
        B[1 : l.n_sites, 1 : p.particles] = A[1 : l.n_sites, 1 : p.particles]
        B[l.n_sites+1 : p.N, p.particles+1 : p.particles + p.n_B] = eye(p.n_B)
    else
        B[:] = 0
        B[1 : p.n_A, 1 : p.particles] = A[1 : p.n_A, 1 : p.particles]
        B[p.n_A+1 : l.n_sites, p.particles+1 : p.particles+p.n_B] = eye(p.n_B)
        B[l.n_sites+1 : p.N, 1 : p.particles] = A[p.n_A+1 : l.n_sites, 1 : p.particles]
    end
end

function enlarge_thinized(A::Array{greens_type, 2}, B::Array{greens_type, 2}, replica::Int, p::parameters, l::lattice)
    N_eff = p.particles + p.n_B
    if replica == 1
        B[:] = 0.
        B[1 : l.n_sites, 1 : p.particles] = A[1 : l.n_sites, 1 : p.particles]
        B[l.n_sites+1 : p.N, p.particles+1 : p.particles + p.n_B] = eye(p.n_B)
        B[1 : l.n_sites, p.particles+p.n_B+1 : p.N] = A[1:l.n_sites, p.particles+1:l.n_sites]
    else
        B[:] = 0
        B[1 : p.n_A, 1 : p.particles] = A[1 : p.n_A, 1 : p.particles]
        B[p.n_A+1 : l.n_sites, p.particles+1 : p.particles+p.n_B] = eye(p.n_B)
        B[l.n_sites+1 : p.N, 1 : p.particles] = A[p.n_A+1 : l.n_sites, 1 : p.particles]
        B[1 : p.n_A, N_eff+1 : p.N] = A[1 : p.n_A, p.particles+1 : l.n_sites]
        B[l.n_sites+1 : p.N, N_eff+1:p.N] = A[p.n_A+1:l.n_sites, p.particles+1 : l.n_sites]
    end
end

function enlarge_thinized(A::Array{real_type, 1}, B::Array{real_type, 1}, replica::Int, p::parameters, l::lattice)
    B[:] = 1.
    N_eff = p.particles + p.n_B
    B[1 : p.particles] = A[1 : p.particles]
    B[p.particles + p.n_B + 1 : end] = A[p.particles+1 : end]
end

################################################################################
# Calculation of Green's functions
################################################################################

function calculate_greens_UDT(s::stack, U, D, T)
    B = \(transpose(T), conj(U))
    C = transpose(B) + diagm(D)
    Q_p, R_p, p_p = qr(C, Val{true}; thin=false);
    p_p_T = copy(p_p); p_p_T[p_p] = collect(1:length(p_p))
    Qs = transpose(conj(U * Q_p))
    Rs = R_p[:, p_p_T] * T
    s.greens = \(Rs, Qs)
    println("UDT done")
    # println(diag(s.greens))
end


function calculate_greens_full(s::stack, p::parameters, l::lattice)
    Us_inv = zeros(greens_type, 3 * l.n_sites, 3 * l.n_sites)
    Ts_inv = copy(Us_inv)
    M = zeros(greens_type, 3 * l.n_sites, 3 * l.n_sites)

    if s.direction == 1
        Ul = col_to_invertible(s.Ul, p, l)
        Dl = ones(l.n_sites) * 1e-32
        Dl[1:p.particles] = s.Dl
        Tl = eye(greens_type, l.n_sites, l.n_sites)
        Tl[1:p.particles, 1:p.particles] = s.Tl

        Tr_T = col_to_invertible(s.Ur, p, l)
        Dr = ones(l.n_sites) * 1e-32
        Dr[1:p.particles] = s.Dr
        Ur_T = eye(greens_type, l.n_sites, l.n_sites)
        Ur_T[1:p.particles, 1:p.particles] = s.Tr
        # Um  Dm  Tm  Ul  Dl  Tl Ur  Dr Tr
        # Ul  Dl  Tl  Ur  Dr  Tr Tm' Dm Um'
        M[1:1 * l.n_sites, 1:l.n_sites] = ctranspose(s.u_temp) * conj(Tr_T)
        M[l.n_sites + 1:2 * l.n_sites, l.n_sites + 1:2 * l.n_sites] = inv(transpose(Ur_T)) * inv(Tl)
        M[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = ctranspose(Ul) * inv(s.t_temp)

        M[1:l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = diagm(s.d_temp)
        M[1 * l.n_sites + 1:2 * l.n_sites, 1:l.n_sites] = -diagm(Dr)
        M[2 * l.n_sites + 1:3 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = -diagm(Dl)

        U_l, D_l, T_l = decompose_udt(M)
        s.det = sum(log(abs(D_l)))
        # println("As expected: ", logabsdet(U_l))
        # println("As expected: ", logabsdet(T_l))

        Us_inv[1:l.n_sites, 1:l.n_sites] = ctranspose(s.u_temp)
        Us_inv[1 * l.n_sites + 1:2 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = ctranspose(Ul)
        Us_inv[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = inv(transpose(Ur_T))

        Ts_inv[1:l.n_sites, 1:l.n_sites] = conj(Tr_T)
        Ts_inv[1 * l.n_sites + 1:2 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = inv(Tl)
        Ts_inv[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = inv(s.t_temp)

        g_large = (Ts_inv * inv(T_l)) * (spdiagm(1./D_l) * (ctranspose(U_l) * Us_inv))
        s.greens = g_large[1:l.n_sites, 1:l.n_sites]

    elseif s.direction == -1
        Ul = col_to_invertible(s.Ul, p, l)
        Dl = ones(l.n_sites) * 1e-32
        Dl[1:p.particles] = s.Dl
        Tl = eye(Complex{Float64}, l.n_sites, l.n_sites)
        Tl[1:p.particles, 1:p.particles] = s.Tl

        Tr_T = col_to_invertible(s.Ur, p, l)
        Dr = ones(l.n_sites) * 1e-32
        Dr[1:p.particles] = s.Dr
        Ur_T = eye(greens_type, l.n_sites, l.n_sites)
        Ur_T[1:p.particles, 1:p.particles] = s.Tr
        # Ul Dl Tl Ur Dr Tr Tm' Dm Um'
        M[1:1 * l.n_sites, 1:l.n_sites] = ctranspose(Ul) * conj(s.u_temp)
        M[l.n_sites + 1:2 * l.n_sites, l.n_sites + 1:2 * l.n_sites] = inv(transpose(s.t_temp)) * inv(transpose(Tr_T))
        M[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = inv(transpose(Ur_T)) * inv(Tl)

        M[1:l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = diagm(Dl)
        M[1 * l.n_sites + 1:2 * l.n_sites, 1:l.n_sites] = -diagm(s.d_temp)
        M[2 * l.n_sites + 1:3 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = -diagm(Dr)

        U_l, D_l, T_l = decompose_udt(M)
        s.det = sum(log(abs(D_l)))

        Us_inv[1:l.n_sites, 1:l.n_sites] = ctranspose(Ul)
        Us_inv[1 * l.n_sites + 1:2 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = inv(transpose(Ur_T))
        Us_inv[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = inv(transpose(s.t_temp))

        # greens.determinant += logabsdet(Us_inv)

        Ts_inv[1:l.n_sites, 1:l.n_sites] = conj(s.u_temp)
        Ts_inv[1 * l.n_sites + 1:2 * l.n_sites, 1 * l.n_sites + 1:2 * l.n_sites] = conj(Ur_T)
        Ts_inv[2 * l.n_sites + 1:3 * l.n_sites, 2 * l.n_sites + 1:3 * l.n_sites] = inv(Tl)

        # greens.determinant += logabsdet(Ts_inv)

        g_large = (Ts_inv * inv(T_l)) * (spdiagm(1./D_l) * (ctranspose(U_l) * Us_inv))
        s.greens = g_large[1:l.n_sites, 1:l.n_sites]
        # println(diag(s.greens))
    else
        error("Not a valid direction")
    end
end

function calculate_greens_full(s_A::stack, s_B::stack, p::parameters, l::lattice)

    Us_inv = zeros(greens_type, 5 * p.N, 5 * p.N)
    Ts_inv = copy(Us_inv)
    M = zeros(Complex{Float64}, 5 * p.N, 5 * p.N)

    actv_rep   = 1
    inactv_rep = 2

    lU1_T = zeros(greens_type, p.N, p.N)
    lU2 = zeros(greens_type, p.N, p.N)
    lU3_T = zeros(greens_type, p.N, p.N)
    lU4 = zeros(greens_type, p.N, p.N)
    lU5_T = zeros(greens_type, p.N, p.N)

    lD1 = zeros(real_type, p.N)
    lD2 = zeros(real_type, p.N)
    lD3 = zeros(real_type, p.N)
    lD4 = zeros(real_type, p.N)
    lD5 = zeros(real_type, p.N)

    lT1_T = eye(greens_type, p.N)
    lT2 = eye(greens_type, p.N)
    lT3_T = eye(greens_type, p.N)
    lT4 = eye(greens_type, p.N)
    lT5_T = eye(greens_type, p.N)


    if s_A.direction == 1

        U4 = col_to_invertible(s_B.Ul, p, l)
        D4 = ones(l.n_sites) * 1e-32
        D4[1:p.particles] = s_B.Dl
        T4 = eye(Complex{Float64}, l.n_sites, l.n_sites)
        T4[1:p.particles, 1:p.particles] = s_B.Tl

        T3_T = col_to_invertible(s_B.Ur, p, l)
        D3 = ones(l.n_sites) * 1e-32
        D3[1:p.particles] = s_B.Dr
        U3_T = eye(Complex{Float64}, l.n_sites, l.n_sites)
        U3_T[1:p.particles, 1:p.particles] = s_B.Tr

        U2 = col_to_invertible(s_A.Ul, p, l)
        D2 = ones(l.n_sites) * 1e-32
        D2[1:p.particles] = s_A.Dl
        T2 = eye(Complex{Float64}, l.n_sites, l.n_sites)
        T2[1:p.particles, 1:p.particles] = s_A.Tl

        T1_T = col_to_invertible(s_A.Ur, p, l)
        D1 = ones(l.n_sites) * 1e-32
        D1[1:p.particles] = s_A.Dr
        U1_T = eye(Complex{Float64}, l.n_sites, l.n_sites)
        U1_T[1:p.particles, 1:p.particles] = s_A.Tr

        enlarge(s_A.u_temp, s_A.u_large, actv_rep, p, l)
        enlarge_thinized(T1_T, s_A.t_large, actv_rep, p, l)
        # display(s_A.t_large); println("\n")
        M[0 * p.N + 1:1 * p.N, 0 * p.N + 1:1 * p.N] = ctranspose(s_A.u_large) * conj(s_A.t_large)
        display(M[0 * p.N + 1:1 * p.N, 0 * p.N + 1:1 * p.N]); println("\n")

        Us_inv[1:p.N, 1:p.N] = ctranspose(s_A.u_large)
        Ts_inv[1:p.N, 1:p.N] = conj(s_A.t_large)

        enlarge(U1_T, s_A.u_large, actv_rep, p, l)
        enlarge(T2, s_A.t_large, actv_rep, p, l)

        M[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(transpose(s_A.u_large)) * inv(s_A.t_large)
        Us_inv[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(transpose(s_A.u_large))
        Ts_inv[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(s_A.t_large)

        enlarge_thinized(U2, s_A.u_large, actv_rep, p, l)
        enlarge_thinized(T3_T, s_A.t_large, inactv_rep, p, l)

        M[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = ctranspose(s_A.u_large) * conj(s_A.t_large)
        Us_inv[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = ctranspose(s_A.u_large)
        Ts_inv[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = conj(s_A.t_large)

        enlarge(U3_T, s_A.u_large, inactv_rep, p, l)
        enlarge(T4, s_A.t_large, inactv_rep, p, l)

        M[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = inv(transpose(s_A.u_large)) * inv(s_A.t_large)
        Us_inv[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = inv(transpose(s_A.u_large))
        Ts_inv[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = inv(s_A.t_large)

        enlarge_thinized(U4, s_A.u_large, inactv_rep, p, l)
        enlarge(s_A.t_temp, s_A.t_large, actv_rep, p, l)

        M[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = ctranspose(s_A.u_large) * inv(s_A.t_large)
        Us_inv[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = ctranspose(s_A.u_large)
        Ts_inv[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = inv(s_A.t_large)

        enlarge(s_A.d_temp, s_A.d_large, actv_rep, p, l)
        M[1:p.N, 4 * p.N + 1:5 * p.N] = diagm(s_A.d_large)
        enlarge_thinized(D1, s_A.d_large, actv_rep, p, l)
        M[1 * p.N + 1:2 * p.N, 0 * p.N + 1:1 * p.N] = -diagm(s_A.d_large)
        enlarge_thinized(D2, s_A.d_large, actv_rep, p, l)
        M[2 * p.N + 1:3 * p.N, 1 * p.N + 1:2 * p.N] = -diagm(s_A.d_large)
        enlarge_thinized(D3, s_A.d_large, inactv_rep, p, l)
        M[3 * p.N + 1:4 * p.N, 2 * p.N + 1:3 * p.N] = -diagm(s_A.d_large)
        enlarge_thinized(D4, s_A.d_large, inactv_rep, p, l)
        M[4 * p.N + 1:5 * p.N, 3 * p.N + 1:4 * p.N] = -diagm(s_A.d_large)

        U_l, D_l, T_l = decompose_udt(M)
        s_A.AB_det = sum(log(abs(D_l)))
        g_large = (Ts_inv * inv(T_l)) * (spdiagm(1./D_l) * (ctranspose(U_l) * Us_inv))

        s_A.AB_greens = g_large[1:p.N, 1:p.N]

    elseif s_A.direction == -1

        U4 = col_to_invertible(s_A.Ul, p, l)
        D4 = ones(l.n_sites) * 1e-32
        D4[1:p.particles] = s_A.Dl
        T4 = eye(Complex{Float64}, l.n_sites, l.n_sites)
        T4[1:p.particles, 1:p.particles] = s_A.Tl

        enlarge_thinized(U4, lU4, actv_rep, p, l)
        enlarge_thin(D4, lD4, actv_rep, p, l)
        lT4[1:p.particles, 1:p.particles] = s_A.Tl

        T3_T = col_to_invertible(s_A.Ur, p, l)
        D3 = ones(l.n_sites) * 1e-32
        D3[1:p.particles] = s_A.Dr
        U3_T = eye(Complex{Float64}, l.n_sites, l.n_sites)
        U3_T[1:p.particles, 1:p.particles] = s_A.Tr

        lU3_T[1:p.particles, 1:p.particles] = s_A.Tr
        enlarge_thin(D3, lD3, actv_rep, p, l)
        enlarge_thinized(U3_T, lU3_T, actv_rep, p, l)

        U2 = col_to_invertible(s_B.Ul, p, l)
        D2 = ones(l.n_sites) * 1e-32
        T2 = eye(Complex{Float64}, l.n_sites, l.n_sites)
        T2[1:p.particles, 1:p.particles] = s_B.Tl

        enlarge_thinized(U2, lU2, inactv_rep, p, l)
        enlarge_thin(D2, lD2, inactv_rep, p, l)
        lT2[1:p.particles, 1:p.particles] = s_B.Tl

        T1_T = col_to_invertible(s_B.Ur, p, l)
        D1 = ones(l.n_sites) * 1e-32
        U1_T = eye(Complex{Float64}, l.n_sites, l.n_sites)
        U1_T[1:p.particles, 1:p.particles] = s_B.Tr

        lU1_T[1:p.particles, 1:p.particles] = s_B.Tr
        enlarge_thin(D1, lD1, inactv_rep, p, l)
        enlarge_thinized(U1_T, lU1_T, inactv_rep, p, l)

        enlarge(s_A.u_temp, T5_T, actv_rep, p, l)
        enlarge(s_A.d_temp, D5  , actv_rep, p, l)
        enlarge(s_A.t_temp, U5_T, actv_rep, p, l)

        enlarge_thinized(U4, s_A.u_large, actv_rep, p, l)
        # enlarge(U4, s_A.u_large, actv_rep, p, l)
        enlarge(s_A.u_temp, s_A.t_large, actv_rep, p, l)
        # println("1")
        # display(s_A.u_large); println("\n")
        # display(s_A.t_large); println("\n")

        M[0 * p.N + 1:1 * p.N, 0 * p.N + 1:1 * p.N] = ctranspose(s_A.u_large) * conj(s_A.t_large)
        # display(M[0 * p.N + 1:1 * p.N, 0 * p.N + 1:1 * p.N]); println("\n")
        Us_inv[1:p.N, 1:p.N] = ctranspose(s_A.u_large)
        Ts_inv[1:p.N, 1:p.N] = conj(s_A.t_large)

        enlarge_thin(s_A.t_temp, s_A.u_large, actv_rep, p, l)
        enlarge_thinized(T1_T, s_A.t_large, inactv_rep, p, l)
        # println("2")
        # display(s_A.u_large); println("\n")
        # display(s_A.t_large); println("\n")

        M[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(transpose(s_A.u_large)) * inv(transpose(s_A.t_large))
        # display(M[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N]); println("\n")
        Us_inv[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(transpose(s_A.u_large))
        Ts_inv[1 * p.N + 1:2 * p.N, 1 * p.N + 1:2 * p.N] = inv(transpose(s_A.t_large))

        enlarge_thin(U1_T, s_A.u_large, inactv_rep, p, l)
        enlarge_thin(T2, s_A.t_large, inactv_rep, p, l)
        # println("3")
        # display(real(s_A.u_large)); println("\n")
        # display(s_A.t_large); println("\n")

        M[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = conj(s_A.u_large) * inv(s_A.t_large)
        # display(M[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N]); println("\n")
        Us_inv[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = conj(s_A.u_large)
        Ts_inv[2 * p.N + 1:3 * p.N, 2 * p.N + 1:3 * p.N] = inv(s_A.t_large)

        enlarge_thinized(U2, s_A.u_large, inactv_rep, p, l)
        # enlarge(U2, s_A.u_large, inactv_rep, p, l)
        enlarge_thinized(T3_T, s_A.t_large, actv_rep, p, l)
        # display(s_A.u_large); println("\n")
        # display(s_A.t_large); println("\n")
        M[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = ctranspose(s_A.u_large) * inv(transpose(s_A.t_large))
        # display(M[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N]); println("\n")
        Us_inv[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = ctranspose(s_A.u_large)
        Ts_inv[3 * p.N + 1:4 * p.N, 3 * p.N + 1:4 * p.N] = inv(transpose(s_A.t_large))

        enlarge_thin(U3_T, s_A.u_large, actv_rep, p, l)
        enlarge_thin(T4, s_A.t_large, actv_rep, p, l)
        # display(s_A.u_large); println("\n")
        # display(s_A.t_large); println("\n")
        M[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = conj(s_A.u_large) * inv(s_A.t_large)
        Us_inv[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = conj(s_A.u_large)
        Ts_inv[4 * p.N + 1:5 * p.N, 4 * p.N + 1:5 * p.N] = inv(s_A.t_large)

        enlarge_thinized(D4, s_A.d_large, actv_rep, p, l)
        M[1:p.N, 4 * p.N + 1:5 * p.N] = diagm(s_A.d_large)

        enlarge(s_A.d_temp, s_A.d_large, actv_rep, p, l)
        M[1 * p.N + 1:2 * p.N, 0 * p.N + 1:1 * p.N] = -diagm(s_A.d_large)

        enlarge_thinized(D1, s_A.d_large, inactv_rep, p, l)
        M[2 * p.N + 1:3 * p.N, 1 * p.N + 1:2 * p.N] = -diagm(s_A.d_large)

        enlarge_thinized(D2, s_A.d_large, inactv_rep, p, l)
        M[3 * p.N + 1:4 * p.N, 2 * p.N + 1:3 * p.N] = -diagm(s_A.d_large)

        enlarge_thinized(D3, s_A.d_large, actv_rep, p, l)
        M[4 * p.N + 1:5 * p.N, 3 * p.N + 1:4 * p.N] = -diagm(s_A.d_large)

        U_l, D_l, T_l = decompose_udt(M)
        s_A.AB_det = sum(log(abs(D_l)))
        # println(s_A.AB_det)

        g_large = (Ts_inv * inv(T_l)) * (spdiagm(1./D_l) * (ctranspose(U_l) * Us_inv))
        s_A.AB_greens = g_large[1:p.N, 1:p.N]
        # println(real(diag(s_A.AB_greens)))
        display(real(s_A.AB_greens)); println("\n")
        println("Greens is done")
    else
        error("Not a valid direction")
    end
end

################################################################################
# Propagation
################################################################################

function propagate(s::stack_type, p::parameter_type, l::lattice)
    # println("Propagate ", s.direction, " - ", s.current_slice)
    # println("In propagate\t", s.d_stack[:, end])
    if s.direction == 1
        if mod(s.current_slice, p.safe_mult) == 0
            s.current_slice += 1
            if s.current_slice == 1
                s.u_temp = eye(greens_type, l.n_sites)
                s.d_temp = ones(real_type, l.n_sites)
                s.t_temp = eye(greens_type, l.n_sites)

                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]

                calculate_greens_full(s, p, l)
            elseif s.current_slice > 1 && s.current_slice < Int(p.slices / 2)
                idx = Int((s.current_slice - 1) / p.safe_mult)
                # println("Now working with $(idx)")
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, s.n_elements - idx ], s.d_stack_r[:, s.n_elements - idx ], s.t_stack_r[:, :, s.n_elements - idx]
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]

                add_slice_sequence_left_temp(s, idx, p, l)

                s.greens_temp[:] = s.greens[:]
                s.greens_temp[:] = multiply_hopping_matrix_right(s.greens_temp, s, p, l, -1.)
                s.greens_temp[:] = multiply_hopping_matrix_left(s.greens_temp, s, p, l, 1.)

                calculate_greens_full(s, p, l)
                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t+1 Propagation stability\t", diff)
                end
            else
                s.u_stack_r[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack_r[:, 1] = ones(real_type, size(s.d_stack_r)[1])
                s.t_stack_r[:, :, 1] = eye(greens_type, size(s.d_stack_r)[1], size(s.d_stack_r)[1])

                for i in reverse(2:s.n_elements)
                    add_slice_sequence_right(s, i, p, l)
                end
                s.direction = -1
                s.current_slice = p.slices + 1
                propagate(s, p, l)
            end
        else
            s.current_slice += 1
            if s.curr_interaction == s.n_interactions
                s.greens[:] = multiply_hopping_matrix_right(s.greens, s, p, l, -1.)
                s.greens[:] = multiply_hopping_matrix_left(s.greens, s, p, l, 1.)
            end
        end
    elseif s.direction == -1
        if mod(s.current_slice - 1, p.safe_mult) == 0
            s.current_slice -= 1
            idx = Int(s.current_slice / p.safe_mult) + 2 - s.n_elements
            if s.current_slice == p.slices

                s.u_temp = eye(greens_type, l.n_sites)
                s.d_temp = ones(real_type, l.n_sites)
                s.t_temp = eye(greens_type, l.n_sites)

                s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]

                calculate_greens_full(s, p, l)

                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)

            elseif s.current_slice > p.slices / 2 && s.current_slice < p.slices
                s.greens_temp[:] = s.greens[:]
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack_l[:, :, idx], s.d_stack_l[:, idx], s.t_stack_l[:, :, idx]
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]
                add_slice_sequence_right_temp(s, idx, p, l)
                calculate_greens_full(s, p, l)
                diff = maximum(diag(abs(s.greens_temp - s.greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t-1  Propagation stability\t", diff)
                end

                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
            elseif s.current_slice == p.slices / 2
                # println("Rebuilding stack")
                s.u_stack_l[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack_l[:, 1] = ones(real_type, size(s.d_stack_l)[1])
                s.t_stack_l[:, :, 1] = eye(greens_type, size(s.d_stack_l)[1], size(s.d_stack_l)[1])

                for i in s.n_elements:length(s.ranges)
                    add_slice_sequence_left(s, i, p, l)
                end

                s.direction = 1
                s.current_slice = 0
                propagate(s, p, l)
            end
        else
            s.current_slice -= 1
            if s.curr_interaction == 1
                s.greens = multiply_hopping_matrix_left(s.greens, s, p, l, -1.)
                s.greens = multiply_hopping_matrix_right(s.greens, s, p, l, 1.)
            end
        end
    end
end


function propagate(s::stack_type, s_B::stack_type, p::parameter_type, l::lattice)
    # println("Propagate ", s.direction, " - ", s.current_slice)
    # println("In propagate\t", s.d_stack[:, end])
    if s.direction == 1
        if mod(s.current_slice, p.safe_mult) == 0
            s.current_slice += 1
            if s.current_slice == 1
                # println("At the top")
                s.Ul, s.Dl, s.Tl = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]
                s.Ur, s.Dr, s.Tr = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]
                s_B.Ul, s_B.Dl, s_B.Tl = s_B.u_stack_l[:, :, end], s_B.d_stack_l[:, end], s_B.t_stack_l[:, :, end]
                s_B.Ur, s_B.Dr, s_B.Tr = s_B.u_stack_r[:, :, end], s_B.d_stack_r[:, end], s_B.t_stack_r[:, :, end]

                s.u_large = eye(greens_type, p.N)
                s.d_large = ones(greens_type, p.N)
                s.t_large = eye(greens_type, p.N)

                s.u_temp = eye(greens_type, l.n_sites)
                s.d_temp = ones(real_type, l.n_sites)
                s.t_temp = eye(greens_type, l.n_sites)

                calculate_greens_full(s, s_B, p, l)

            elseif s.current_slice > 1 && s.current_slice < Int(p.slices / 2)
                idx = Int((s.current_slice - 1) / p.safe_mult)
                # println("Now working with $(idx)")
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, s.n_elements - idx ], s.d_stack_r[:, s.n_elements - idx ], s.t_stack_r[:, :, s.n_elements - idx]
                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]

                add_slice_sequence_left_temp(s, idx, p, l)

                s.AB_greens_temp[:] = s.AB_greens[:]
                s.AB_greens_temp[:] = multiply_hopping_matrix_right(s.AB_greens_temp, s, p, l, -1.)
                s.AB_greens_temp[:] = multiply_hopping_matrix_left(s.AB_greens_temp, s, p, l, 1.)

                calculate_greens_full(s, s_B, p, l)

                diff = maximum(diag(abs(s.AB_greens_temp - s.AB_greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t+1 Propagation stability\t", diff)
                end
            else
                s.u_stack_r[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack_r[:, 1] = ones(real_type, p.particles)
                s.t_stack_r[:, :, 1] = eye(greens_type, p.particles, p.particles)

                for i in reverse(2:s.n_elements)
                    add_slice_sequence_right(s, i, p, l)
                end
                s.direction = -1
                s.current_slice = p.slices + 1
                propagate(s, s_B, p, l)
            end
        else
            s.current_slice += 1
            if s.curr_interaction == s.n_interactions
                s.AB_greens[:] = multiply_hopping_matrix_right(s.AB_greens, s, p, l, -1.)
                s.AB_greens[:] = multiply_hopping_matrix_left(s.AB_greens, s, p, l, 1.)
            end
        end
    elseif s.direction == -1
        if mod(s.current_slice - 1, p.safe_mult) == 0
            s.current_slice -= 1
            idx = Int(s.current_slice / p.safe_mult) + 2 - s.n_elements
            if s.current_slice == p.slices
                s.Ul, s.Dl, s.Tl = s.u_stack_l[:, :, end], s.d_stack_l[:, end], s.t_stack_l[:, :, end]
                s.Ur, s.Dr, s.Tr = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]
                s_B.Ul, s_B.Dl, s_B.Tl = s_B.u_stack_l[:, :, end], s_B.d_stack_l[:, end], s_B.t_stack_l[:, :, end]
                s_B.Ur, s_B.Dr, s_B.Tr = s_B.u_stack_r[:, :, end], s_B.d_stack_r[:, end], s_B.t_stack_r[:, :, end]

                s.u_large = eye(greens_type, p.N)
                s.d_large = ones(greens_type, p.N)
                s.t_large = eye(greens_type, p.N)

                s.u_temp = eye(greens_type, l.n_sites)
                s.d_temp = ones(real_type, l.n_sites)
                s.t_temp = eye(greens_type, l.n_sites)

                calculate_greens_full(s, s_B, p, l)
                # println("At the top ", real(diag(s.AB_greens)))
                # display(s.AB_greens); println("\n")

                tmp = s.hopping_matrix_AB_inv * s.AB_greens
                s.AB_greens = tmp * s.hopping_matrix_AB

            elseif s.current_slice > p.slices / 2 && s.current_slice < p.slices
                # println("Idx is $(idx)")

                s.AB_greens_temp[:] = s.AB_greens[:]

                s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack_l[:, :, idx], s.d_stack_l[:, idx], s.t_stack_l[:, :, idx]
                s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack_r[:, :, end], s.d_stack_r[:, end], s.t_stack_r[:, :, end]

                add_slice_sequence_right_temp(s, idx, p, l)

                calculate_greens_full(s, s_B, p, l)

                # println("Here's da old G\t", real(diag(s.AB_greens_temp)))
                # println("Here's da new G\t", real(diag(s.AB_greens)))

                diff = maximum(diag(abs(s.AB_greens_temp - s.AB_greens)))
                if diff > 1e-4
                    println(s.current_slice, "\t-1  Propagation stability\t", diff)
                end

                tmp = s.hopping_matrix_AB_inv * s.AB_greens
                s.AB_greens = tmp * s.hopping_matrix_AB

            elseif s.current_slice == p.slices / 2
                # println("Rebuilding stack")
                s.u_stack_l[:, :, 1] = get_wavefunction(s, p, l)
                s.d_stack_l[:, 1] = ones(real_type, p.particles)
                s.t_stack_l[:, :, 1] = eye(greens_type, p.particles, p.particles)

                for i in s.n_elements:length(s.ranges)
                    add_slice_sequence_left(s, i, p, l)
                end

                s.direction = 1
                s.current_slice = 0
                propagate(s, s_B, p, l)
            end
        else
            s.current_slice -= 1
            if s.curr_interaction == 1
                # println("hmm $(s.current_slice)")
                tmp = s.hopping_matrix_AB_inv * s.AB_greens
                s.AB_greens = tmp * s.hopping_matrix_AB
            end
        end
    end
end
