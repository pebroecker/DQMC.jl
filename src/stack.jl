function initialize_stack(s::stack_type, p::parameter_type, l::lattice)
    s.n_elements = convert(Int, p.slices / p.safe_mult) + 1
    println("There are $(s.n_elements) elements in the stack")
    println("We have matrices of size $(l.n_sites) x $(p.particles)")

    s.u_stack = zeros(greens_type, l.n_sites, p.particles, s.n_elements)
    s.d_stack = zeros(real_type, p.particles, s.n_elements)
    s.t_stack = zeros(greens_type, p.particles, p.particles, s.n_elements)

    s.greens = zeros(greens_type, l.n_sites, l.n_sites)
    s.greens_temp = zeros(greens_type, l.n_sites, l.n_sites)
    s.hopping_matrix = zeros(greens_type, l.n_sites, l.n_sites)
    s.hopping_matrix_inv = zeros(greens_type, l.n_sites, l.n_sites)

    s.Ul = zeros(greens_type, l.n_sites, p.particles)
    s.Tl = eye(greens_type, p.particles, p.particles)
    s.Ur = zeros(greens_type, l.n_sites, p.particles)
    s.Tr = eye(greens_type, p.particles, p.particles)
    s.Dl = ones(real_type, p.particles)
    s.Dr = ones(real_type, p.particles)

    s.U = zeros(greens_type, l.n_sites, l.n_sites)
    s.Q = zeros(greens_type, l.n_sites, l.n_sites)
    s.D = zeros(real_type, p.particles)
    s.R = zeros(greens_type, p.particles, p.particles)
    s.T = zeros(greens_type, p.particles, p.particles)

    s.ranges = UnitRange[]

    for i in 1:s.n_elements - 1
        push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
    end
end


function build_stack(s::stack_type, p::parameter_type, l::lattice)
    s.u_stack[:, :, 1] = get_wavefunction(s, p, l)
    s.d_stack[:, 1] = ones(real_type, size(s.d_stack)[1])
    s.t_stack[:, :, 1] = eye(greens_type, size(s.d_stack)[1], size(s.d_stack)[1])

    println(size(s.u_stack))
    println(size(s.d_stack))
    println(size(s.t_stack))
    
    for i in 1:length(s.ranges)
        add_slice_sequence_left(s, i, p, l)
    end

    s.current_slice = p.slices + 1
    s.direction = -1
end

function multiply_hopping_matrix_left(M::Array{greens_type, 2}, s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return get_hopping_matrix(s, p, l, pref) * M
end

function multiply_hopping_matrix_right(M::Array{greens_type, 2},
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return M * get_hopping_matrix(s, p, l, pref)
end

function multiply_slice_matrix_left(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return slice_matrix(slice, s, p, l, pref) * M
end

function multiply_slice_matrix_right(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return M * slice_matrix(slice, s, p, l, pref)
end
