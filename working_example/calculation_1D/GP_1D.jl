
##################################################
## 1D evolution of a given potential/initial state
##################################################


include("../utilities.jl")
using Statistics
using DelimitedFiles


# multidimensional parameter space may be defined within that function and then looped over with linear index
function evolution_GP_MPS_1D(ind)


    function GP_Trotter_1D(psi0, potentials, R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)

        # constants
        fourier_tol = 1e-10
        exp_pot_tol = 1e-8
        beta = 2 # parameter for FD distribution for momentum space low pass

        println("m = ", m)
        println("tol = ", tol)
        println("g = ", g)
        println("dt = ", dt)

        # construct initial quantics MPS with explicitly set pivot
        qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
        localdims = fill(2, R)
        qf(x) = psi0(QG.quantics_to_origcoord(qgrid, x))
        cf = TCI.CachedFunction{ComplexF64}(qf, localdims)
        p1 = ones(Int, length(localdims)) # get pivots in non-trivial branches
        p1[1] = 2
        #p2 = 2 .* ones(Int, length(localdims))
        pivots = [p1]
        psi_mps, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots, tolerance=tol)
        println("psi_mps ", psi_mps)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))

        # construct potential MPOs:
        sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
        pot_mpos_IT = [] 
        pot_half_mpos_IT = []
        for i in 1:length(potentials)
            exp_potential_mpo = exp_potential_MPO(potentials[i], dt, xmin, xmax, R, exp_pot_tol)
            exp_potential_mpo_half = exp_potential_MPO(potentials[i], dt/2, xmin, xmax, R, exp_pot_tol)
            println("exp_potential_mpo ", exp_potential_mpo)
            println("exp_potential_mpo_half ", exp_potential_mpo_half)
            exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
            exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)
            push!(pot_mpos_IT, exp_potential_mpo_IT)
            push!(pot_half_mpos_IT, exp_potential_mpo_half_IT)
        end
        
        # full MPO for kinetic evolution, FT⁻1 x low-pass x exp(Laplacian) x FT
        exp_kinetic_mpo = get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)

        # create directories for output and save wavefunction
        remove_directory("Runs/Data_tmp")
        remove_directory("Runs/Data_tmp/Plots")
        remove_directory("Runs/Data_tmp/Data_reconstructed")
        mkdir("Runs/Data_tmp")
        mkdir("Runs/Data_tmp/Plots")
        mkdir("Runs/Data_tmp/Data_reconstructed")
        save_object("Runs/Data_tmp/psi_mps_1.jld2", psi_mps)

        # do second-order Trotter evolution
        for i in 1:length(potentials)
            psi_mps = apply_MPO_IT(psi_mps, pot_half_mpos_IT[i], sites_m, maxdim)
            println("norm pot = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
            println("norm ftt = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

        for j in 1:Nsteps
            if j%10 == 0
                println("doing step j = ", j)
            end

            psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
            println("norm kinetic = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            psi_mps = normalise(psi_mps, R, xmin, xmax)

            if g!=0
                psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt, tol, maxdim)
                println("norm ftt = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            end
            for i in 1:length(potentials)
                psi_mps = apply_MPO_IT(psi_mps, pot_mpos_IT[i], sites_m, maxdim)
                println("norm pot = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            end

            # save every 10th MPS
            if j%10 == 0
                save_object("Runs/Data_tmp/psi_mps_$(j).jld2", psi_mps)
            end

        end
        psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
            println("norm f_tt = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end
        for i in 1:length(potentials)
            psi_mps = apply_MPO_IT(psi_mps, pot_half_mpos_IT[i], sites_m, maxdim)
            println("norm pot = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

        # rename saved data, name each run e.g. with parameters used or some other sensible name
        #file_name = "_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A2_$(A2)_A3_$(A3)"
        file_name = "_test_run"
        mv("Runs/Data_tmp", "Runs/wf_evolution_"*join(string.(potentials), "_")*file_name)

	# some dummy return values, remnant from performance calculations where the individual parameter point are saved in dataframe
        return (R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)
    end


    ###########################
    # parameters of calculation
    ###########################

    # hard parameters
    m = 1
    tol = 1e-10

    # parameters of evolution
    Rs = [30] # spatial resolution as 2^R
    dts = [0.01] # time increment
    Ns = [1000] # number of individual Trotter steps -> total evolution time T = N*dt
    maxdims = [14] # max bond dimension in MPO applications
    gs = [5] # non-linearity 
    x_widths = [500.] # width of the box as [-x_width, x_width]

    # parameters of potentials
    omegas1 = [0.01] # different spatial oscillations/widths, as defined below
    omegas2 = [10]
    omegas3 = [1e5]
    A1s = [80] # different amplitudes, as defined below
    A2s = [5.0]
    A3s = [20.0]

    # construct parameter space as power set of individual parameter sets above 
    parameter_space = collect(Iterators.product(Rs, dts, Ns, maxdims, omegas1, omegas2, omegas3, x_widths, gs, A1s, A2s, A3s))
    ps = parameter_space[ind]
    R = ps[1]
    dt = ps[2]
    Nsteps = ps[3]
    maxdim = ps[4]
    omega1 = ps[5]
    omega2 = ps[6]
    omega3 = ps[7]
    x_width = ps[8]
    g = ps[9]
    A1 = ps[10]
    A2 = ps[11]
    A3 = ps[12]
    
    # remnant derived parameter from incommensurate evolution example
    sigma = sqrt(1/(sqrt(2*A2)*omega2)) # according to Taylor'd harmonic potential
    println("sigma = ", sigma)
    println("parameter space: ", ps)


    ##########################
    # different initial states
    ##########################
    
    function gaussian(x)
        return (1/pi)^(1/4) * exp(-x^2/2)
    end

    function gaussian_localised(x)
        return (1/(pi*sigma^2))^(1/4) * exp(-x^2/(2*sigma^2))
    end

    function gaussian_even_odd_comb(x)
        n_max = 50
        gaussian_sum = 0.0 + 0.0im
        for n in -n_max:n_max
            center = n * 2π
            gaussian_sum += gaussian_localised(x - center)
        end
        return gaussian_sum
    end


    ######################
    # different potentials
    ######################
    
    function quadratic_potential(x)
        return omega1 * x^2
    end

    function quadratic_potential_mod(x)
        return omega1 * x^2 + A1*sin(omega2*x)^2
    end

    function quadratic_potential_double_mod(x)
        return omega1 * x^2 + A1*sin(omega2*x)^2 + A2*sin(omega3*x)^2
    end

    function quadratic_potential_double_mod_cos(x)
        return omega1 * x^2 + A1*cos(omega2*x) + A2*cos(omega3*x)
    end

    function zero_potential(x)
        return 0
    end

    function sine_potential2(x)
        return A2*sin(omega2*x)^2
    end

    function sine_potential3(x)
        return A3*sin(omega3*x)^2
    end

    function double_mod_potential(x)
        return A1*sin(omega1*x)^2 + A2*sin(omega2*x)^2
    end

    #############
    # calculation 
    #############
    
    # specify initial state function as well as list of potentials!
    meta_data = GP_Trotter_1D(gaussian, [quadratic_potential, sine_potential2], R, -x_width, x_width, g, dt, Nsteps, m, tol, maxdim)
    println("meta_data ", meta_data)

end

# execute, can also easily index different points in the total parameter space
evolution_GP_MPS_1D(1)

