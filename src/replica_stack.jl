function initialize_stack(s::stack_type, p::parameter_type, l::lattice)
    if p.stack_handling == "ground_state"
        s.n_elements = convert(Int, p.slices / (2 * p.safe_mult)) + 1
    else
        s.n_elements = convert(Int, p.slices / (p.safe_mult)) + 1
    end

    println("There are $(s.n_elements) elements in the stack")
    println("We have matrices of size $(l.n_sites) x $(p.particles)")

    s.u_stack_l = zeros(greens_type, l.n_sites, p.particles, s.n_elements)
    s.d_stack_l = zeros(real_type, p.particles, s.n_elements)
    s.t_stack_l = zeros(greens_type, p.particles, p.particles, s.n_elements)

    s.u_stack_r = zeros(greens_type, l.n_sites, p.particles, s.n_elements)
    s.d_stack_r = zeros(real_type, p.particles, s.n_elements)
    s.t_stack_r = zeros(greens_type, p.particles, p.particles, s.n_elements)

    s.u_temp = eye(greens_type, l.n_sites)
    s.d_temp = ones(real_type, l.n_sites)
    s.t_temp = eye(greens_type, l.n_sites)

    s.u_large = eye(greens_type, p.N)
    s.d_large = ones(real_type, p.N)
    s.t_large = eye(greens_type, p.N)

    s.greens = zeros(greens_type, l.n_sites, l.n_sites)
    s.greens_temp = zeros(greens_type, l.n_sites, l.n_sites)
    s.hopping_matrix = zeros(greens_type, l.n_sites, l.n_sites)
    s.hopping_matrix_inv = zeros(greens_type, l.n_sites, l.n_sites)

    s.AB_greens = zeros(greens_type, p.N, p.N)
    s.AB_greens_temp = zeros(greens_type, p.N, p.N)

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

    for i in 1:(s.n_elements - 1) * 2
        push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
    end
end


function build_stack(s::stack_type, p::parameter_type, l::lattice)
    s.u_stack_l[:, :, 1] = get_wavefunction(s, p, l)
    s.d_stack_l[:, 1] = ones(real_type, size(s.d_stack_l)[1])
    s.t_stack_l[:, :, 1] = eye(greens_type, size(s.d_stack_l)[1], size(s.d_stack_l)[1])

    # display(s.u_stack_l[:, :, 1]); println("\n")

    for i in s.n_elements:length(s.ranges)
        add_slice_sequence_left(s, i, p, l)
    end

    s.u_stack_r[:, :, 1] = get_wavefunction(s, p, l)
    s.d_stack_r[:, 1] = ones(real_type, size(s.d_stack_r)[1])
    s.t_stack_r[:, :, 1] = eye(greens_type, size(s.d_stack_r)[1], size(s.d_stack_r)[1])

    # display(s.u_stack_r[:, :, 1]); println("\n")

    for i in reverse(2:s.n_elements)
        add_slice_sequence_right(s, i, p, l)
    end

    s.current_slice = p.slices + 1
    s.direction = -1

    println("Stack is built")
end


function add_slice_sequence_left(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack_l[:, :, idx + 1 - s.n_elements])

    for slice in s.ranges[idx]
        slice_mat = slice_matrix(slice, s, p, l, 1.)
        curr_U = slice_mat * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack_l[:, idx + 1 - s.n_elements])
    s.u_stack_l[:, :, idx + 2 - s.n_elements], s.d_stack_l[:, idx + 2 - s.n_elements], T = decompose_udt(curr_U)
    s.t_stack_l[:, :, idx + 2 - s.n_elements] = T * s.t_stack_l[:, :, idx + 1 - s.n_elements]
    # println("d stack left")
    # display(s.d_stack_l[:, idx + 2 - s.n_elements]); println("\n")
    # println("Putting $(s.ranges[idx]) into idx $(idx + 2 - s.n_elements)")
end

function add_slice_sequence_right(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_stack_r[:, :, s.n_elements - idx + 1])

    for slice in reverse(s.ranges[idx - 1])
        slice_mat = slice_matrix(slice, s, p, l, 1.)
        curr_U = transpose(slice_mat) * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_stack_r[:, s.n_elements - idx + 1])
    s.u_stack_r[:, :, s.n_elements - idx + 2], s.d_stack_r[:, s.n_elements - idx + 2], T = decompose_udt(curr_U)
    s.t_stack_r[:, :, s.n_elements - idx + 2] = T * s.t_stack_r[:, :, s.n_elements - idx + 1]
    # println("d stack right")
    # display(s.d_stack_l[:, s.n_elements - idx + 2]); println("\n")
    # println("Putting $(s.ranges[idx - 1]) into idx $(s.n_elements + 2 - idx)")
end


function add_slice_sequence_left_temp(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_temp)
    # println("Recalculating $(s.ranges[idx])")
    for slice in s.ranges[idx]
        slice_mat = slice_matrix(slice, s, p, l, 1.)
        curr_U = slice_mat * curr_U
    end

    curr_U = curr_U * spdiagm(s.d_temp)
    s.u_temp, s.d_temp, T = decompose_udt(curr_U)
    s.t_temp = T * s.t_temp
end


function add_slice_sequence_right_temp(s::stack_type, idx::Int, p::parameter_type, l::lattice)
    curr_U = copy(s.u_temp)
    # println("Recalculating right temp ", s.ranges[idx + s.n_elements - 1])
    for slice in reverse(s.ranges[idx + s.n_elements - 1])
        slice_mat = slice_matrix(slice, s, p, l, 1.)
        curr_U = transpose(slice_mat) * curr_U
    end

    curr_U =  curr_U * spdiagm(s.d_temp)
    s.u_temp, s.d_temp, T = decompose_udt(curr_U)
    s.t_temp = T * s.t_temp
end


function multiply_hopping_matrix_left(M::Array{greens_type, 2}, s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    if size(M, 1) == l.n_sites
        return get_hopping_matrix(s, p, l, pref) * M
    else
        return get_hopping_matrix_AB(s, p, l, pref) * M
    end
end

function multiply_hopping_matrix_right(M::Array{greens_type, 2},
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    if size(M, 1) == l.n_sites
        return M * get_hopping_matrix(s, p, l, pref)
    else
        return M * get_hopping_matrix_AB(s, p, l, pref)
    end
end

function multiply_slice_matrix_left(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return slice_matrix(slice, s, p, l, pref) * M
end

function multiply_slice_matrix_right(M::Array{greens_type, 2}, slice::Int,
    s::stack_type, p::parameter_type, l::lattice, pref::Float64=1.)
    return M * slice_matrix(slice, s, p, l, pref)
end
