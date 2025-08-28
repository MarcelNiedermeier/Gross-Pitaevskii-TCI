
##################################################
## 2D evolution of a given potential/initial state
##################################################

include("../utilities.jl")



###########
# functions
###########

function fidelity_ITensor_2D(psi1, psi2, R, xmin, xmax, ymin, ymax)
    dx = (xmax-xmin)/2^R
    dy = (ymax-ymin)/2^R

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:2*R]

    #transforming to ITensors
    IT_psi1 = MPS(psi1; sites=sites_m)
    IT_psi2 = MPS(psi2; sites=sites_m)
    return abs(ITensors.inner(IT_psi1, IT_psi2)*dx*dy)^2
end


function normalise_2D(psi, R, xmin, xmax, ymin, ymax)
    n = sqrt(sqrt(fidelity_ITensor_2D(psi, psi, R, xmin, xmax, ymin, ymax)))
    println("norm squared =", n^2)
    n_R = n^(1/(2*R))
    tensor_list = []
    for i in 1:2*R
        push!(tensor_list, 1/n_R*psi[i])
    end
    return TCI.TensorTrain([tensor_list[i] for i in 1:2*R])
end



function GP_Trotter_MPS_2D(psi0, potentials, R, xmin, xmax, ymin, ymax, g, dt, Nsteps, m, tol, maxdim)

    # constants
    M = 2^R
    kcut = 2^8
    beta = 2
    #cont_method = :TCI
    cont_method = :naive
    fourier_tol = 1e-8

    # construct initial quantics MPS directly
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, xygrid; tolerance=tol)
    psi_mps = psi.tci
    println("psi_mps ", psi_mps)
    
    println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
    println("renormalising")
    psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
    println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
    
    # save wavefunctions -> adjust details in path as needed
    path = "Data_Gaussian_moving_k5_"*join(string.(potentials), "_")*"_$(R)_g_$(g)_xy_$(xmax)_Nsteps_$(Nsteps)_maxdim_$(maxdim)_tol_$(tol)_dt_$(dt)"
    remove_directory("Runs/"*path)
    remove_directory("Runs/"*path*"/Plots")
    remove_directory("Runs/"*path*"/Data_reconstructed")
    mkdir("Runs/"*path)
    mkdir("Runs/"*path*"/Plots")
    mkdir("Runs/"*path*"/Data_reconstructed")
    save_object("Runs/"*path*"/psi_mps_1.jld2", psi_mps)

    # construct Laplacian MPO in momentum space
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:2*R]
    exp_lap_momentum_mpo = exp_lap_Fourier_MPO_lowpass_2D(xmin, xmax, ymin, ymax, R, dt, m, tol, kcut, beta)
    exp_lap_momentum_mpo_IT = IT_MPO_conversion(exp_lap_momentum_mpo, sites_m)
    println("exp_lap_momentum_mpo ", exp_lap_momentum_mpo)  
    
    # construct all potential MPOs
    pot_mpos_IT = []
    pot_half_mpos_IT = []
    for i in 1:length(potentials)
    	exp_potential_mpo = exp_potential_MPO_2D(potentials[i], dt, xmin, xmax, ymin, ymax, R, tol)
    	exp_potential_mpo_half = exp_potential_MPO_2D(potentials[i], dt/2, xmin, xmax, ymin, ymax, R, tol)
    	println("exp_potential_mpo ", exp_potential_mpo)
        println("exp_potential_mpo_half ", exp_potential_mpo_half)
        exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
        exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)
        push!(pot_mpos_IT, exp_potential_mpo_IT)
        push!(pot_half_mpos_IT, exp_potential_mpo_half_IT)
    end

    # do second-order Trotter evolution
    for i in 1:length(potentials)
        psi_mps = apply_MPO_IT(psi_mps, pot_half_mpos_IT[i], sites_m, maxdim)
    end
    if g != 0
        psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt/2, tol)
    end

    for j in 1:Nsteps
        if j%1 == 0
            println("doing step j = ", j)
        end

	# kinetic evolution
        ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
        println("ft_psi_mps ", ft_psi_mps)
        ft_psi_mps = apply_MPO_IT(ft_psi_mps, exp_lap_momentum_mpo_IT, sites_m, maxdim)
        psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)
        
        # renormalise
        println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
        println("renormalising")
        psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
        println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))

        # potential evolution
        for i in 1:length(potentials)
	    psi_mps = apply_MPO_IT(psi_mps, pot_mpos_IT[i], sites_m, maxdim)
	end
        if g != 0
            psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt, tol)
        end
        println("apply_f_tt_2D ", psi_mps)       
        
        # save every 10th MPS
        if j%10 == 0
            save_object("Runs/"*path*"/psi_mps_$(j).jld2", psi_mps)
        end
    end

    ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
    ft_psi_mps = apply_MPO_IT(ft_psi_mps, exp_lap_momentum_mpo_IT, sites_m, maxdim)
    psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)
    
    println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
    println("renormalising")
    psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
    println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))

    # half potential evolution
    for i in 1:length(potentials)
        psi_mps = apply_MPO_IT(psi_mps, pot_half_mpos_IT[i], sites_m, maxdim)
    end
    if g != 0
        psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt/2, tol)
    end

    # some dummy return values
    return (R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)
end




##############
# calculations
##############

function gaussian_2D(x, y)
    return (1/pi)^(1/4) * exp(-(x^2+y^2)/2)
end

function gaussian_moving_2D_k5(x, y)
    k = 5
    return (1/pi)^(1/4) * exp(-(x^2+y^2)/2) * exp(1.0im*k*x)
end


function sine_2D(x, y)
    return sin(x)*sin(y)
end


function quadratic_potential_2D(x, y)
    return 0.001*(x^2+y^2)
end

function quadratic_potential_aniso_2D(x, y)
    return 0.01*x^2+0.015*y^2
end


function sine_potential_modulation_x_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*sin(omega2*x)^2
end

function sine_potential_modulation_y_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*sin(omega2*y)^2
end

function sine_potential_modulation_twist1_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*sin(omega2/sqrt(2) * (x+y))^2 
end

function sine_potential_modulation_twist2_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*sin(omega2/sqrt(2) * (y-x))^2 
end


function sine_potential_modulation_2D(x, y)
    A2 = 10
    omega2 = 10
    return A2*(sin(omega2*x)^2 + sin(omega2*y)^2)
end

function sine_potential_modulation_twist_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*(sin(omega2/sqrt(2) * (x+y))^2 + sin(omega2/sqrt(2) * (y-x))^2)
end


function sine_potential_modulation_eightfold_2D(x, y)
    A2 = 10
    omega2 = 1
    return A2*(sin(omega2*x)^2 + sin(omega2/sqrt(2) * (x+y))^2 + sin(omega2/sqrt(2) * (y-x))^2 + sin(omega2*y)^2)
end


function quadratic_potential_aniso_anharmonic_2D(x, y)
    return 0.01*x^2+0.015*y^2 + 0.012*x*y
end

function zero_potential_2D(x, y)
    return 0
end

function sine_potential_2D(x, y)
    return 1 - sin(x)^2 * sin(y)^2
end

# parameters
R = 20 # system discretisation as 2^R x 2^R
xmin, xmax = -100., 100. # box size
ymin, ymax = -100., 100.
tol = 1e-8 # per tensor tolerance
dt = 0.01 # time increment
g = 5 # non-linearity
Nsteps = 20 # number of Trotter steps, such that total time T = Nsteps x dt
m = 1 # mass 
maxdim = 50 # bond dim cutoff for MPO applications

data = GP_Trotter_MPS_2D(gaussian_moving_2D_k5, [quadratic_potential_2D, sine_potential_modulation_x_2D, sine_potential_modulation_y_2D, sine_potential_modulation_twist1_2D, sine_potential_modulation_twist2_2D], R, xmin, xmax, ymin, ymax, g, dt, Nsteps, m, tol, maxdim)

println("data = ", data)




