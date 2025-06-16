
##################################################
## 1D evolution of a given potential/initial state
##################################################


###################
## Test for 1D case
###################

include("../utilities.jl")
using Statistics
#using IterTools
#using CSV
using DelimitedFiles


function evolution_GP_MPS_1D(ind)


    function GP_Trotter_MPS_reduced_eval_1D(psi0, potentials, R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)

        # constants
        fourier_tol = 1e-10
        exp_pot_tol = 1e-8
        beta = 2

        println("m = ", m)
        println("tol = ", tol)
        println("g = ", g)
        println("dt = ", dt)

        # construct initial quantics MPS
        qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)

        localdims = fill(2, R)
        qf(x) = psi0(QG.quantics_to_origcoord(qgrid, x))
        cf = TCI.CachedFunction{ComplexF64}(qf, localdims)
        p1 = ones(Int, length(localdims)) # get pivots in non-trivial branches
        p1[1] = 2
        #p2 = 2 .* ones(Int, length(localdims))
        pivots = [p1]
        psi_mps, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots, tolerance=tol)

        #psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol, nrandominitpivot=1000)
        #psi_mps = psi.tci
        println("psi_mps ", psi_mps)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))

        # construct relevant MPOs:
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
        #exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, exp_pot_tol)
        #exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, exp_pot_tol)
        exp_kinetic_mpo = get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)
        #sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
        #exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
        #exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)
        #println("exp_potential_mpo ", exp_potential_mpo)
        #println("exp_kinetic_mpo ", exp_kinetic_mpo)

        # save wavefunction
        remove_directory("Runs/Data_tmp")
        remove_directory("Runs/Data_tmp/Plots")
        remove_directory("Runs/Data_tmp/Data_reconstructed")
        mkdir("Runs/Data_tmp")
        mkdir("Runs/Data_tmp/Plots")
        mkdir("Runs/Data_tmp/Data_reconstructed")
        save_object("Runs/Data_tmp/psi_mps_1.jld2", psi_mps)

        # do second-order Trotter evolution
        #psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
        for i in 1:length(potentials)
            psi_mps = apply_MPO_IT(psi_mps, pot_half_mpos_IT[i], sites_m, maxdim)
            println("norm pot = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end
        #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
            println("norm ftt = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        end
        # (psi, R, xmin, xmax, g, dt, tol)
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
            # renormalise
            if j%10 == 0
                #println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
                println("renormalising")
            end
            psi_mps = normalise(psi_mps, R, xmin, xmax)
            if j%10 == 0
                println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
                println("psi_mps = ", psi_mps)
            end


            if g!=0
                psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt, tol, maxdim)
                println("norm ftt = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            end
            for i in 1:length(potentials)
                psi_mps = apply_MPO_IT(psi_mps, pot_mpos_IT[i], sites_m, maxdim)
                println("norm pot = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            end
            #psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

            # renormalise
            #if j%1 == 0
            #    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            #    println("renormalising")
            #end
            #psi_mps = normalise(psi_mps, R, xmin, xmax)
            #if j%1 == 0
            #    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            #    println("psi_mps = ", psi_mps)
            #end

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
        #psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
        #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

        # rename saved data
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*join(string.(potentials), "_")*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A2_$(A2)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*join(string.(potentials), "_")*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O2_$(omega2)_A2_$(A2)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_even_comb_"*join(string.(potentials), "_")*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O2_$(omega2)_A2_$(A2)")
        mv("Runs/Data_tmp", "Runs/wf_evolution_"*join(string.(potentials), "_")*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A2_$(A2)_A3_$(A3)")

        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A1_$(A1)_A2_$(A2)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_A1_$(A1)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_A2_$(A2)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_sigma_$(sigma)")
        #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_sigma_$(sigma)")

        return (R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)
    end



    # hard parameters
    m = 1
    tol = 1e-10

    # parameters
    Rs = [30]
    dts = [0.01]
    Ns = [5000]#100, 500, 1000, 5000, 10000]
    maxdims = [14]
    #gs = [0, 1, 10]
    #gs = [0, 1, 5]
    gs = [5]
    x_widths = [500.]

    # optical trap etc
    #omegas1 = [1/10]#, 1/50, 1/100, 1/200, 1/500]
    omegas1 = [0.01]#, 0.02]
    #omegas1 = [3*1]
    #omegas1 = [3*1e5]
    #omegas2 = [1e3, 1e4, 1e5]#, 50, 100, 200, 500, 1000, 5000, 10000]
    omegas2 = [10]
    #omegas2 = [1e4]#, 50, 100, 200, 500, 1000, 5000, 10000]
    #omegas2 = [1/sqrt(2)]
    omegas3 = [1e5]
    A1s = [80]
    A2s = [5.0]
    A3s = [20.0]

    # incommensurate AAH
    #omegas2 = [1.]
    #omegas3 = [(1+sqrt(5))/2 * 1]
    #omegas3 = [1/3 * 1]
    #A1s = [10, 20, 50, 100]
    #A1s = [80]
    #A2s = [3.0]
    #A3s = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]#, 2, 3]
    #A3s = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #A3s = [0.8]#, 2, 3]

    #A = 10
    #A1 = 100
    #A2 = 5
    #sigma = 1 # choose two orders of magnitude smaller than potential trough
    #sigma = 1e-7 # choose two orders of magnitude smaller than potential trough

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
    sigma = sqrt(1/(sqrt(2*A2)*omega2)) # according to Taylor'd harmonic potential
    println("sigma = ", sigma)
    println("parameter space: ", ps)

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

    # calculation
    #GP_Trotter_MPS_reduced_eval_1D
    #wf_evolution, _, widths = GP_Trotter_MPS_1D(gaussian, quadratic_potential, R, -x_width, x_width, g, dt, Nsteps, m, tol, prec, kcut, maxdim, true, true)
    #wf_evolution, wf_red, _, widths = GP_Trotter_MPS_reduced_eval_1D(gaussian, quadratic_potential_double_mod, R, -x_width, x_width, x0_red, xreds, g, dt, Nsteps, m, tol, prec, kcut, maxdim, true, true)

    #meta_data = GP_Trotter_MPS_reduced_eval_1D(gaussian_even_odd_comb, [sine_potential2, sine_potential3], R, -x_width, x_width, g, dt, Nsteps, m, tol, maxdim)
    meta_data = GP_Trotter_MPS_reduced_eval_1D(gaussian, [sine_potential2, sine_potential3], R, -x_width, x_width, g, dt, Nsteps, m, tol, maxdim)
    #meta_data = GP_Trotter_MPS_reduced_eval_1D(gaussian, [quadratic_potential, sine_potential2, sine_potential3], R, -x_width, x_width, g, dt, Nsteps, m, tol, maxdim)
    #meta_data = GP_Trotter_MPS_reduced_eval_1D(gaussian, [quadratic_potential, sine_potential2], R, -x_width, x_width, g, dt, Nsteps, m, tol, maxdim)
    println("meta_data ", meta_data)

    #gfkkf


    # output
    #writedlm("Data/wf_evolution_quad_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1).txt", wf_evolution, ",")
    #writedlm("Data/wf_evolution_quad_double_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A1_$(A1)_A2_$(A2).txt", wf_evolution, ",")
    #for i in 1:length(xreds)
        #writedlm("Data/wf_evolution_red_$(x0_red)_$(xreds[i])_quad_double_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A1_$(A1)_A2_$(A2).txt", wf_red[i], ",")
        #writedlm("Data/wf_evolution_red_$(x0_red)_$(xreds[i])_quad_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1).txt", wf_red[i], ",")
    #end
    #writedlm("Data/wf_evolution_red_$(x0_red)_$(xred2)_quad_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A_$(A).txt", wf_evolution_red2, ",")
    #writedlm("Data/width_quad_double_mod_MPS_1D_R_$(R)_Nsteps_dt_$(dt)_$(Nsteps)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A1_$(A1)_A2_$(A2).txt", widths, ",")
    #writedlm("Data/width_quad_mod_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1).txt", widths, ",")

end

evolution_GP_MPS_1D(1)
#evolution_GP_MPS_1D(2)
#evolution_GP_MPS_1D(3)
#evolution_GP_MPS_1D(4)

#for i in 1:6
#    evolution_GP_MPS_1D(i)
#end
