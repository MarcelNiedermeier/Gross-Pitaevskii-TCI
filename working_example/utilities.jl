
############
## Utilities
############


using LaTeXStrings
using Plots
gr()

import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate, quanticsfouriermpo
import TensorCrossInterpolation as TCI
using TensorOperations


import TCIITensorConversion
using ITensors
using ITensorMPS
import Quantics: fouriertransform, Quantics

using JLD2


##########################
# general helper functions
##########################

""" Function to remove a given directory if it exists. """
function remove_directory(directoryname::String)

    # build bash file
    touch("remove_directory.sh")
    remove = open("remove_directory.sh", "w")

    write(remove, "#!/bin/bash \n")
    write(remove, "if [ -d "*directoryname*" ]; then rm -r "*directoryname*"; fi \n")
    close(remove)

    run(`bash remove_directory.sh`)
    rm("remove_directory.sh")
end


""" Function to create the temporary directories where the data
is saved. """
function create_data_directories()

    # remove previous directories if they (still) exist
    remove_directory("Data_temporary")
    remove_directory("outfiles")

    # create needed directories
    mkdir("Data_temporary")
    mkdir("outfiles")
end



function get_max_rank(tt)
    ranks = []
    for i in 1:length(tt)
        #println("check size: ", size(tt[i])[3])
        push!(ranks, size(tt[i])[3])
    end
    return maximum(ranks)
end


function build_TT(tensors)
    return TCI.TensorTrain(tensors)
end


function TT_to_MPO(tt)
    """ Convert a quantics tensor train to an MPO for element-wise multiplication. """
    tensor_list = []
    delta = Complex.(zeros((2, 2, 2)))
    delta[1, 1, 1] = 1.
    delta[2, 2, 2] = 1.
    for i in 1:length(tt)
        tmp_tensor = Complex.(zeros((size(tt[i])[1], 2, 2, size(tt[i])[3])))
        @tensor begin
            tmp_tensor[a, e, f, c] = tt[i][a, b, c]*delta[e, f, b]
        end
        #println("temp tensor ", size(tmp_tensor), typeof(tmp_tensor))
        push!(tensor_list, tmp_tensor)
    end
    return TCI.TensorTrain([tensor_list[i] for i in 1:length(tt)])
end


function IT_MPO_conversion(tt, sites)
    N = length(tt)
    localdims = TCI.sitedims(tt)
    linkdims = [1, TCI.linkdims(tt)..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N]

    tensors_ = [ITensor(deepcopy(tt[n]), links[n], sites[n], prime(sites[n]), links[n + 1]) for n in 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return ITensorMPS.MPO(tensors_)
end



#############
# 1 dimension
#############



#function lap_Fourier(k, R, xmin, xmax)
#    L = xmax - xmin
#    M = 2^R
#    return -4*M^2/L^2 * sin(pi/M * k)^2
#end


function evaluate_wavefunction(wf, qgrid, prec, xmax)
    N_prec = 2^prec
    maxindex = QG.origcoord_to_grididx(qgrid, xmax)
    testindices = Int.(round.(LinRange(1, maxindex, N_prec)))
    return [TCI.evaluate(wf, QG.grididx_to_quantics(qgrid, i)) for i in testindices]
end


function evaluate_wavefunction_reduced(wf, qgrid, prec, xmin, xmax)
    N_prec = 2^prec
    minindex = QG.origcoord_to_grididx(qgrid, xmin)
    maxindex = QG.origcoord_to_grididx(qgrid, xmax)
    testindices = Int.(round.(LinRange(minindex, maxindex, N_prec)))
    return [TCI.evaluate(wf, QG.grididx_to_quantics(qgrid, i)) for i in testindices]
end


function pos_MPO(xmin, xmax, R, tol)
    function pos(x)
        return x
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, pos, qgrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function pos_squared_MPO(xmin, xmax, R, tol)
    function pos_squared(x)
        return x^2
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, pos_squared, qgrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function exp_potential_MPO(pot, h, xmin, xmax, R, tol)
    function exp_pot(x)
        return exp(-1.0im * pot(x) * h)
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_pot, qgrid, tolerance=tol, maxiter=2000, nrandominitpivot=1000, sweepstrategy=:forward)
    return TT_to_MPO(ci.tci)
end


#function func_MPO(func, xmin, xmax, R, tol)
#    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
#    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, func, qgrid, tolerance=tol)
#    return TT_to_MPO(ci.tci)
#end


function exp_lap_Fourier_MPO(xmin, xmax, R, dt, m, tol)
    M = 2^R
    kcut = 1*(M-1)
    function lap_Fourier(k)
        L = xmax - xmin
        return exp(1.0im/2* (-4)*M^2/(m*L^2) * sin(pi/M * k)^2 *dt)
    end
    kgrid = QG.DiscretizedGrid{1}(R, 0, kcut; includeendpoint=true)
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, lap_Fourier, kgrid, maxbonddim=20)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, lap_Fourier, kgrid, tolerance=tol)
    println(ci.tci)
    return TT_to_MPO(ci.tci)
end


function exp_lap_Fourier_MPO_lowpass(xmin, xmax, R, dt, m, tol, kcut, beta)
    M = 2^R
    kmax = 2^R - 1
    L = xmax - xmin
    function lap_Fourier_lowpass(k)
        return (1 + 1/(exp((k-kcut)*beta) + 1) - 1/(exp((k - (kmax-kcut))*beta) + 1)) * exp(1.0im/2* (-4)*M^2/(m*L^2) * sin(pi/M * k)^2 *dt)
        #return exp(1.0im/2* (-4)*M^2/(m*L^2) * sin(pi/M * k)^2 *dt * exp(-beta*sin(pi/M * k)^2))
    end
    kgrid = QG.DiscretizedGrid{1}(R, 0, kmax; includeendpoint=true)
    localdims = fill(2, R)
    qf(k) = lap_Fourier_lowpass(QG.quantics_to_origcoord(kgrid, k))
    cf = TCI.CachedFunction{ComplexF64}(qf, localdims)

    # get pivots in non-trivial branches
    p1 = ones(Int, length(localdims))
    p2 = 2 .* ones(Int, length(localdims))
    pivots = [p1, p2]

    ci, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots)
    return TT_to_MPO(ci)
end


function get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)

    # get initial MPOs
    fourier_tol = 1e-12
    exp_lap_tol = 1e-10
    qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=fourier_tol)
    invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=fourier_tol)

    # get a Gaussian trial state
    function gaussian_trial(x)
        return (1/pi)^(1/4) * exp(-x^2/2)
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, gaussian_trial, qgrid, tolerance=tol, nrandominitpivot=1000)
    psi_mps = psi.tci
    println("psi mps initial ", psi_mps)

    n = 0.5
    kcut = 8

    while n < 0.99
        println(" try kcut = ", kcut)

        # construct kinetic operator
        exp_lap_momentum_mpo_low_pass_contr = exp_lap_Fourier_MPO_lowpass(xmin, xmax, R, dt, m, exp_lap_tol, 2^kcut, beta)
        println("exp_lap_momentum_mpo_low_pass_contr ", exp_lap_momentum_mpo_low_pass_contr)
        op1 = TCI.contract(TCI.reverse(exp_lap_momentum_mpo_low_pass_contr), qftmpo; algorithm=:naive, tolerance=tol, maxbonddim=100)
        global op2 = TCI.contract(TCI.reverse(invqftmpo), op1; algorithm=:naive, tolerance=tol, maxbonddim=100)
        println("op2 ", op2)
        psi_lowpass_back_mps = TCI.contract(op2, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=100)
        println("psi_lowpass_back_mps ", psi_lowpass_back_mps)

        # check loss of norm
        n = fidelity_ITensor(psi_lowpass_back_mps, psi_lowpass_back_mps, R, xmin, xmax)
        kcut += 1
        println("n = ", n)
        println("exp_lap_momentum_mpo_low_pass_contr ", exp_lap_momentum_mpo_low_pass_contr)
    end
    #return exp_lap_momentum_mpo_low_pass_contr
    return op2
end


function low_pass_MPO_FD(R, tol, kcut, beta)
    kmax = 2^R-1
    function low_pass(k)
        return 1 + 1/(exp((k-kcut)*beta) + 1) - 1/(exp((k - (kmax-kcut))*beta) + 1)
    end
    kgrid = QG.DiscretizedGrid{1}(R, 0, kmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, low_pass, kgrid, tolerance=1e-15, maxiter=1000, sweepstrategy=:forward)
    return TT_to_MPO(ci.tci)
end


function apply_f_tt(psi, R, xmin, xmax, g, dt, tol, maxdim)
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    function exp_nonlinear(x)
        ind = QG.origcoord_to_grididx(qgrid, x)
        wf = TCI.evaluate(psi, QG.grididx_to_quantics(qgrid, ind))
        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
    end
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, initialpivots, tolerance=tol)#, maxbonddim=maxdim)
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, nrandominitpivot=2^R, tolerance=tol)#, maxbonddim=maxdim)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, tolerance=tol, maxbonddim=maxdim)
    #  ci, ranks, errors = quanticscrossinterpolate(ComplexF64, lap_Fourier_lowpass_2D, kxkygrid, initialpivots, tolerance=tol)
    return ci.tci
end


function apply_f_tt_cutoff(psi, R, xmin, xmax, g, dt, tol)
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    function exp_nonlinear(x)
        ind = QG.origcoord_to_grididx(qgrid, x)
        wf = TCI.evaluate(psi, QG.grididx_to_quantics(qgrid, ind))
        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
    end
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, tolerance=tol)
    return ci.tci
end


#function apply_f_tt_gfunc(psi, g, R, xmin, xmax, g, dt, tol)
#    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
#    function exp_nonlinear(x)
#        ind = QG.origcoord_to_grididx(qgrid, x)
#        wf = TCI.evaluate(psi, QG.grididx_to_quantics(qgrid, ind))
#        return exp(-1.0im*g(x)*abs(wf)^2 * dt)*wf
#    end
#    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, tolerance=tol)
#    return ci.tci
#end

# with initial pivots
#function apply_f_tt(psi, R, xmin, xmax, g, dt, tol, maxdim)
#    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
#    function exp_nonlinear(x)
#        ind = QG.origcoord_to_grididx(qgrid, x)
#        wf = TCI.evaluate(psi, QG.grididx_to_quantics(qgrid, ind))
#        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
#    end
#
#    localdims = fill(2, R)
#    qf(x) = exp_nonlinear(QG.quantics_to_origcoord(qgrid, x))
#    cf = TCI.CachedFunction{ComplexF64}(qf, localdims)
#    
#    pivots =  Vector{Vector{Int64}}()
#    for i in 0:(2^R - 1)
#        # Get the binary representation with R bits
#        bin = reverse(digits(i, base=2, pad=R))
#        # Map 0 → 1 and 1 → 2
#        push!(pivots, [Int64(b+1) for b in bin])
#    end
#    
#    ci, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots, tolerance=tol)
#    return ci
#end



#kgrid = QG.DiscretizedGrid{1}(R, 0, kmax; includeendpoint=true)
#localdims = fill(2, R)
#qf(k) = lap_Fourier_lowpass(QG.quantics_to_origcoord(kgrid, k))
#cf = TCI.CachedFunction{ComplexF64}(qf, localdims)

# get pivots in non-trivial branches
#p1 = ones(Int, length(localdims))
#p2 = 2 .* ones(Int, length(localdims))
#pivots = [p1, p2]

#ci, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots)



function apply_MPO_IT(psi_mps, mpo_IT, sites, maxdim, nsweeps=2)
    #sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    #psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi.tci); sites=sites_m)
    psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi_mps); sites=sites)
    psi_mpo_mps_IT = ITensors.contract(mpo_IT, psi_mps_IT; method="fit", nsweeps=nsweeps, maxdim=maxdim)
    return TCI.TensorTrain(psi_mpo_mps_IT)
end

function apply_MPO_IT_cutoff(psi_mps, mpo_IT, sites, cutoff, nsweeps=2)
    #sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    #psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi.tci); sites=sites_m)
    psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi_mps); sites=sites)
    psi_mpo_mps_IT = ITensors.contract(mpo_IT, psi_mps_IT; method="fit", nsweeps=nsweeps, cutoff=cutoff)
    return TCI.TensorTrain(psi_mpo_mps_IT)
end


#function fidelity(psi, phi, xmin, xmax, prec)
#    dx = (xmax - xmin)/2^prec
#    f = Complex(0)
#    for i in 1:2^prec
#        f += psi[i] * conj(phi[i])
#    end
#    return abs(f*dx)^2
#end


function fidelity_ITensor(psi1, psi2, R, xmin, xmax)
    dx = (xmax-xmin)/2^R

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]

    #transforming to ITensors
    IT_psi1 = MPS(psi1; sites=sites_m)
    IT_psi2 = MPS(psi2; sites=sites_m)
    return abs(ITensors.inner(IT_psi1, IT_psi2)*dx)^2
end


function normalise(psi, R, xmin, xmax)
    n = sqrt(sqrt(fidelity_ITensor(psi, psi, R, xmin, xmax)))
    println("norm squared =", n^2)
    n_R = n^(1/R)
    tensor_list = []
    for i in 1:R
        push!(tensor_list, 1/n_R*psi[i])
    end
    return TCI.TensorTrain([tensor_list[i] for i in 1:R])
end


function expectation_value_ITensor(psi, op, R, xmin, xmax)
    tol = 1e-10
    dx = (xmax-xmin)/2^R

    op_psi = TCI.contract(op, psi; algorithm=:naive, tolerance=tol)

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]

    #transforming to ITensors
    IT_psi = MPS(psi; sites=sites_m)
    IT_op_psi = MPS(op_psi; sites=sites_m)

    return real(ITensors.inner(IT_psi, IT_op_psi)*dx)
    #return abs(ITensors.inner(IT_psi1, IT_psi2)*dx)^2
end


function GP_Trotter_MPS_1D(psi0, pot, R, xmin, xmax, g, dt, Nsteps, m, tol, prec, kcut, maxdim, evaluate_wf=true, calculate_width=true)

    # constants
    #M = 2^R
    #vol_cell = (xmax-xmin)/M
    #xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 10
    fourier_tol = 1e-10
    beta = 2

    println("m = ", m)
    println("tol = ", tol)
    println("g = ", g)
    println("dt = ", dt)

    # save outputs
    bond_dimensions = zeros(Nsteps)
    widths = zeros(Nsteps+1)
    if evaluate_wf
        wf_evolution = Complex.(zeros(Nsteps+1, 2^prec))
    end

    # construct initial quantics MPS
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    psi_mps = psi.tci
    println("psi_mps ", psi_mps)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))

    # construct relevant MPOs:
    exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, tol)
    exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, tol)
    exp_kinetic_mpo = get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)

    #qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=fourier_tol)
    #invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=fourier_tol)
    #exp_lap_momentum_mpo = exp_lap_Fourier_MPO(xmin, xmax, R, dt, m, tol)
    #exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, tol)
    #exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, tol)
    #low_pass_mpo = low_pass_MPO_FD(R, tol, 2^kcut, beta)
    #exp_lap_momentum_mpo_lowpass = TCI.contract(low_pass_mpo, exp_lap_momentum_mpo; algorithm=:naive, tolerance=tol)
    #exp_lap_momentum_mpo_lowpass = exp_lap_Fourier_MPO_lowpass(xmin, xmax, R, dt, m, tol, kcut, beta)
    #exp_lap_momentum_mpo_lowpass = get_exp_Lap_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)
    #op1 = TCI.contract(TCI.reverse(exp_lap_momentum_mpo_lowpass), qftmpo; algorithm=:naive, tolerance=tol, maxbonddim=100)
    #exp_kinetic_mpo = TCI.contract(TCI.reverse(invqftmpo), op1; algorithm=:naive, tolerance=tol, maxbonddim=100)

    #println("exp_lap_momentum_mpo ", exp_lap_momentum_mpo)
    #println("size of exp(Laplacian) [MB] = ", Base.summarysize(exp_lap_momentum_mpo)/10^6)
    #println("low_pass_mpo ", low_pass_mpo)
    #println("exp_lap_momentum_mpo_lowpass ", exp_lap_momentum_mpo_lowpass)
    println("exp_potential_mpo ", exp_potential_mpo)
    println("exp_kinetic_mpo ", exp_kinetic_mpo)

    # MPOs for width
    if calculate_width
        pos_mpo = pos_MPO(xmin, xmax, R, tol)
        pos_sq_mpo = pos_squared_MPO(xmin, xmax, R, tol)
    end

    # save wavefunction
    if evaluate_wf
        wf_evolution[1, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
    end

    if calculate_width
        widths[1] = sqrt(expectation_value_ITensor(psi_mps, pos_sq_mpo, R, xmin, xmax) - expectation_value_ITensor(psi_mps, pos_mpo, R, xmin, xmax)^2)
        println("widths[1] = ", widths[1])
    end

    # do second-order Trotter evolution
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end

    for j in 1:Nsteps
        if j%5 == 0
            println("doing step j = ", j)
        end

        #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
        #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
        psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt, tol, maxdim)
        end
        psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

        if evaluate_wf
            wf_evolution[j+1, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
        end

        if calculate_width
            widths[j+1] = sqrt(expectation_value_ITensor(psi_mps, pos_sq_mpo, R, xmin, xmax) - expectation_value_ITensor(psi_mps, pos_mpo, R, xmin, xmax)^2)
        end

    end
    #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
    #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
    psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("psi_mps = ", psi_mps)

    # maybe return qgrid?
    if evaluate_wf
        return wf_evolution, psi_mps, widths
    else
        return psi_mps, widths
    end
end




function GP_forward_backward_Trotter_MPS_1D(psi0, pot, R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)

    # constants
    #M = 2^R
    #vol_cell = (xmax-xmin)/M
    #xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 10
    fourier_tol = 1e-10
    beta = 2

    println("m = ", m)
    println("tol = ", tol)
    println("g = ", g)
    println("dt = ", dt)

    # save outputs
    #bond_dimensions = zeros(Nsteps)

    # construct initial quantics MPS
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    psi_mps = psi.tci
    println("psi_mps ", psi_mps)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    psi_mps_initial = deepcopy(psi_mps)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))

    # construct relevant MPOs:
    exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, tol)
    exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, tol)
    exp_kinetic_mpo = get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)

    exp_potential_mpo_neg = exp_potential_MPO(pot, -dt, xmin, xmax, R, tol)
    exp_potential_mpo_half_neg = exp_potential_MPO(pot, -dt/2, xmin, xmax, R, tol)
    exp_kinetic_mpo_neg = get_kinetic_lowpass_mpo(xmin, xmax, R, -dt, m, tol, beta)

    # convert to ITensor
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
    exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)
    exp_potential_mpo_neg_IT = IT_MPO_conversion(exp_potential_mpo_neg, sites_m)
    exp_potential_mpo_half_neg_IT = IT_MPO_conversion(exp_potential_mpo_half_neg, sites_m)

    println("exp_potential_mpo ", exp_potential_mpo)
    println("exp_kinetic_mpo ", exp_kinetic_mpo)


    ############################################
    # do second-order Trotter evolution positive
    ############################################

    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end

    for j in 1:Nsteps
        if j%5 == 0
            println("doing step j = ", j)
        end

        psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt, tol, maxdim)
        end
        #psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_IT, sites_m, maxdim)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

    end
    psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("psi_mps = ", psi_mps)


    ############################################
    # do second-order Trotter evolution negative
    ############################################

    #psi_mps = TCI.contract(exp_potential_mpo_half_neg, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_neg_IT, sites_m, maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, -dt/2, tol, maxdim)
    end
    psi_mps = TCI.contract(exp_kinetic_mpo_neg, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

    for j in 1:Nsteps
        if j%5 == 0
            println("doing step j = ", j)
        end

        #psi_mps = TCI.contract(exp_potential_mpo_neg, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_neg_IT, sites_m, maxdim)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, -dt, tol, maxdim)
        end
        psi_mps = TCI.contract(exp_kinetic_mpo_neg, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

        # renormalise
        println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("renormalising")
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
        println("psi_mps = ", psi_mps)

    end

    #psi_mps = TCI.contract(exp_potential_mpo_half_neg, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_neg_IT, sites_m, maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, -dt/2, tol, maxdim)
    end

    # evaluate fidelity between evolved state and initial state
    return fidelity_ITensor(psi_mps, psi_mps_initial, R, xmin, xmax)

    # maybe return qgrid?
    #if evaluate_wf
    #    return wf_evolution, psi_mps, widths
    #else
    #    return psi_mps, widths
    #end

end



function GP_Trotter_MPS_initial_state_1D(psi_mps, pot, R, xmin, xmax, g, dt, Nsteps, m, tol, maxdim)


    # constants
    fourier_tol = 1e-10
    exp_pot_tol = 1e-8
    beta = 2

    println("m = ", m)
    println("tol = ", tol)
    println("g = ", g)
    println("dt = ", dt)

    # construct initial quantics MPS
    #qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)

    #localdims = fill(2, R)
    #qf(x) = psi0(QG.quantics_to_origcoord(qgrid, x))
    #cf = TCI.CachedFunction{ComplexF64}(qf, localdims)
    #p1 = ones(Int, length(localdims)) # get pivots in non-trivial branches
    #p1[1] = 2
    #p2 = 2 .* ones(Int, length(localdims))
    #pivots = [p1]
    #psi_mps, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots, tolerance=tol)

    #psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol, nrandominitpivot=1000)
    #psi_mps = psi.tci
    #println("psi_mps ", psi_mps)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))

    # construct relevant MPOs:
    exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, exp_pot_tol)
    exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, exp_pot_tol)
    exp_kinetic_mpo = get_kinetic_lowpass_mpo(xmin, xmax, R, dt, m, tol, beta)
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
    exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)
    println("exp_potential_mpo ", exp_potential_mpo)
    println("exp_kinetic_mpo ", exp_kinetic_mpo)

    # save wavefunction
    #remove_directory("Runs/Data_tmp")
    #remove_directory("Runs/Data_tmp/Plots")
    #remove_directory("Runs/Data_tmp/Data_reconstructed")
    #mkdir("Runs/Data_tmp")
    #mkdir("Runs/Data_tmp/Plots")
    #mkdir("Runs/Data_tmp/Data_reconstructed")
    #save_object("Runs/Data_tmp/psi_mps_1.jld2", psi_mps)

    # do second-order Trotter evolution
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end

    for j in 1:Nsteps
        if j%10 == 0
            println("doing step j = ", j)
        end

        psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt, tol, maxdim)
        end
        psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_IT, sites_m, maxdim)
        #psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

        # renormalise
        if j%10 == 0
            println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            println("renormalising")
        end
        psi_mps = normalise(psi_mps, R, xmin, xmax)
        if j%10 == 0
            println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
            println("psi_mps = ", psi_mps)
        end

        # save every 10th MPS
        if j%10 == 0
            save_object("Runs/Data_tmp/psi_mps_$(j).jld2", psi_mps)
        end

    end
    psi_mps = TCI.contract(exp_kinetic_mpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol, maxdim)
    end
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

    # renormalise
    println("norm before = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("renormalising")
    psi_mps = normalise(psi_mps, R, xmin, xmax)
    println("norm after = ", fidelity_ITensor(psi_mps, psi_mps, R, xmin, xmax))
    println("psi_mps = ", psi_mps)

    # rename saved data
    #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A1_$(A1)_A2_$(A2)")
    #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_A1_$(A1)")
    #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_A2_$(A2)")
    #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_sigma_$(sigma)")
    #mv("Runs/Data_tmp", "Runs/wf_evolution_"*string(nameof(pot))*"_MPS_1D_R_$(R)_Nsteps_$(Nsteps)_dt_$(dt)_maxdim_$(maxdim)_x_$(x_width)_O1_$(omega1)_O2_$(omega2)_A1_$(A1)_sigma_$(sigma)")

    return psi_mps
end






##################
# error comparison
##################

function GP_Trotter_MPS_1D_comparison(psi0, pot, psi_ana, R, xmin, xmax, g, dt, Nsteps, m, tol, prec, maxdim, save_wf=false)

    # constants
    M = 2^R
    vol_cell = (xmax-xmin)/M
    xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 20
    fourier_tol = 1e-10

    # save outputs
    bond_dimensions = zeros(Nsteps)
    wf_evolution = Complex.(zeros(Nsteps+1, 2^prec))
    error_av = zeros(Nsteps+1)
    error_max = zeros(Nsteps+1)
    fids = zeros(Nsteps+1)

    # construct initial quantics MPS directly
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    psi_mps = psi.tci

    # construct relevant MPOs:
    qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=fourier_tol)
    invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=fourier_tol)
    exp_lap_momentum_mpo = exp_lap_Fourier_MPO(xmin, xmax, R, dt, m, tol)
    exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, tol)
    exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, tol)

    println("exp_lap_momentum_mpo ", exp_lap_momentum_mpo)
    println("exp_potential_mpo ", exp_potential_mpo)

    # check error
    t = 0
    error_av[1] = mean(abs.(evaluate_wavefunction(psi_mps, qgrid, prec, xmax)[:] .- psi_ana(xs, t)[:]))
    error_max[1] = maximum(abs.(evaluate_wavefunction(psi_mps, qgrid, prec, xmax)[:] .- psi_ana(xs, t)[:]))
    fids[1] = fidelity(evaluate_wavefunction(psi_mps, qgrid, prec, xmax), psi_ana(xs, t), xmin, xmax, prec)
    println("fid = ", fids[1])

    # save wavefunction
    if save_wf
        wf_evolution[1, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
    end

    # do second-order Trotter evolution
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol)
    end

    for j in 1:Nsteps
        if j%1 == 0
            println("doing step j = ", j)
        end

        ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
        ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
        psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))

        # half potential evolution
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol)
        end
        psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

        # evaluate wavefunction if desired
        #psi_mps.compress(tol)
        bond_dimensions[j] = get_max_rank(psi_mps)

        # calculate error
        t += dt
        error_av[j+1] = mean(abs.(evaluate_wavefunction(psi_mps, qgrid, prec, xmax)[:] .- psi_ana(xs, t)[:]))
        error_max[j+1] = maximum(abs.(evaluate_wavefunction(psi_mps, qgrid, prec, xmax)[:] .- psi_ana(xs, t)[:]))
        fids[j+1] = fidelity(evaluate_wavefunction(psi_mps, qgrid, prec, xmax), psi_ana(xs, t), xmin, xmax, prec)
        println("fid = ", fids[j+1])
        if fids[j+1] < 1e-3
            break
        end

        # save wavefunction
        if save_wf
            wf_evolution[j+1, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
        end

        # half potential evolution
        if g!=0
            psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol)
        end
        psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

    end
    ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
    ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)
    psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim))
    if g!=0
        psi_mps = apply_f_tt(psi_mps, R, xmin, xmax, g, dt/2, tol)
    end
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=:naive, tolerance=tol, maxbonddim=maxdim)

    # maybe return qgrid?
    if save_wf
        return fids, error_av, error_max, bond_dimensions, wf_evolution
    else
        return fids, error_av, error_max, bond_dimensions
    end
end



##############
# 2 dimensions
##############


function lap_Fourier_2D(kx, ky, R, xmin, xmax)
    L = xmax - xmin
    M = 2^R
    return -4*M^2/L^2 * (sin(pi/M * kx)^2 + sin(pi/M * ky)^2)
end


function evaluate_2D(qtci_xy, R, prec)
    """ Reconstruct the image of a quantics crossinterpolated function in 2D.
    Version for TCI2 object. """
    indexgrid = round.(Int, LinRange(1, 2^R, 2^prec))
    arr = zeros(ComplexF64, 2^prec, 2^prec)
    for i in 1:2^prec
        for j in 1:2^prec
            arr[i, j] = qtci_xy((indexgrid[i], indexgrid[j]))
        end
    end
    return arr
end


function evaluate_2D(qtt_xy, xygrid, R, prec)
    """ Reconstruct the image of a quantics crossinterpolated function in 2D.
    Version for TT object. """
    indexgrid = round.(Int, LinRange(1, 2^R, 2^prec))
    arr = zeros(ComplexF64, 2^prec, 2^prec)
    for i in 1:2^prec
        for j in 1:2^prec
            arr[i, j] = qtt_xy(QG.grididx_to_quantics(xygrid, (indexgrid[i], indexgrid[j])))
        end
    end
    return arr
end


function r_MPO_2D(pot, h, xmin, xmax, ymin, ymax, R, tol)
    function r(x, y)
        return sqrt(x^2 + y^2)
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, r, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function r_squared_MPO_2D(pot, h, xmin, xmax, ymin, ymax, R, tol)
    function r_squared(x, y)
        return x^2 + y^2
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, r_squared, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function exp_potential_MPO_2D(pot, h, xmin, xmax, ymin, ymax, R, tol)
    function exp_pot_2D(x, y)
        return exp(-1.0im * pot(x, y) * h)
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_pot_2D, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function exp_lap_Fourier_MPO_2D(xmin, xmax, ymin, ymax, R, dt, m, tol)
    M = 2^R
    #Mprime = 2^(R-6)
    Mprime = 2^R
    function lap_Fourier_2D(kx, ky)
        L = xmax - xmin
        M = 2^R
        return exp(-1.0im * 2*M^2/(m*L^2) * (sin(pi/M * kx)^2 + sin(pi/M * ky)^2)*dt )
    end
    kxkygrid = QG.DiscretizedGrid{2}(R, (0,0), (Mprime-1, Mprime-1); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, lap_Fourier_2D, kxkygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end



# version with lowpass
function exp_lap_Fourier_MPO_lowpass_2D(xmin, xmax, ymin, ymax, R, dt, m, tol, kcut, beta)
    M = 2^R
    kmax = 2^R - 1
    L = xmax - xmin
    function lap_Fourier_lowpass_2D(kx, ky)
        return (1 + 1/(exp((kx-kcut)*beta) + 1) - 1/(exp((kx - (kmax-kcut))*beta) + 1)) * (1 + 1/(exp((ky-kcut)*beta) + 1) - 1/(exp((ky - (kmax-kcut))*beta) + 1)) * exp(1.0im/2* (-4)*M^2/(m*L^2) * (sin(pi/M * kx)^2 + sin(pi/M * ky)^2)*dt )
    end
    #kgrid = QG.DiscretizedGrid{1}(R, 0, kmax; includeendpoint=true)
    kxkygrid = QG.DiscretizedGrid{2}(R, (0,0), (kmax, kmax); unfoldingscheme=:interleaved)
    #localdims = fill(2, 2*R)
    #qf(kx, ky) = lap_Fourier_lowpass_2D(QG.quantics_to_origcoord(kxkygrid, (kx, ky)))
    #cf = TCI.CachedFunction{ComplexF64}(qf, localdims)

    # get pivots in non-trivial branches
    initialpivots = [[1, 1], [1, 2^R], [2^R, 1], [2^R, 2^R]]

    #ci, ranks, errors = TCI.crossinterpolate2(ComplexF64, cf, localdims, pivots)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, lap_Fourier_lowpass_2D, kxkygrid, initialpivots, tolerance=tol)
    return TT_to_MPO(ci.tci)
end




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




function apply_f_tt_2D(psi, xmin, xmax, ymin, ymax, R, g, dt, tol)
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    function exp_nonlinear_2D(x, y)
        wf = psi(QG.origcoord_to_quantics(xygrid, (x, y)))
        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
    end
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear_2D, xygrid, tolerance=tol)
    return ci.tci
end


function apply_f_tt_2D(psi, xmin, xmax, ymin, ymax, R, g, dt, tol, maxdim)
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    function exp_nonlinear_2D(x, y)
        wf = psi(QG.origcoord_to_quantics(xygrid, (x, y)))
        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
    end
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear_2D, xygrid, tolerance=tol, maxbonddim=maxdim)
    return ci.tci
end



function Fourier_transform_2D(qtt_xy, R; cutoff=1e-10, reverse=true)

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    sites_k = [Index(2, "Qubit,k=$k") for k in 1:R]
    sites_n = [Index(2, "Qubit,n=$n") for n in 1:R]
    sites_l = [Index(2, "Qubit,l=$l") for l in 1:R]

    # make interleaved collection of sites
    sites_mn = collect(Iterators.flatten(zip(sites_m, sites_n)))

    #transforming to ITensors
    IT_mps = MPS(TCI.TensorTrain(qtt_xy); sites=sites_mn)

    # Fourier transform for x, Fourier transform for y
    #IT_mps_FTx = (1/(2sqrt)^R) * fouriertransform(IT_mps; sign=1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    IT_mps_FTx = fouriertransform(IT_mps; sign=1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    #IT_mps_FTx_FTy = (1/sqrt(2)^R) * fouriertransform(IT_mps_FTx; sign=1, tag="n", sitesdst=sites_l, cutoff=cutoff)
    IT_mps_FTx_FTy = fouriertransform(IT_mps_FTx; sign=1, tag="n", sitesdst=sites_l, cutoff=cutoff)

    # reconvert to quantics tt
    if reverse
        return TCI.reverse(TCI.TensorTrain(IT_mps_FTx_FTy))
    else
        return TCI.TensorTrain(IT_mps_FTx_FTy)
    end
end


function inv_Fourier_transform_2D(qtt_xy, R; cutoff=1e-10, reverse=true)

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    sites_k = [Index(2, "Qubit,k=$k") for k in 1:R]
    sites_n = [Index(2, "Qubit,n=$n") for n in 1:R]
    sites_l = [Index(2, "Qubit,l=$l") for l in 1:R]

    # make interleaved collection of sites
    sites_mn = collect(Iterators.flatten(zip(sites_m, sites_n)))

    #transforming to ITensors
    IT_mps = MPS(TCI.TensorTrain(qtt_xy); sites=sites_mn)

    # Fourier transform for x, Fourier transform for y
    #IT_mps_invFTx = (1/sqrt(2)^R) * fouriertransform(IT_mps; sign=-1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    IT_mps_invFTx = fouriertransform(IT_mps; sign=-1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    #IT_mps_invFTx_invFTy = (1/sqrt(2)^R) * fouriertransform(IT_mps_invFTx; sign=-1, tag="n", sitesdst=sites_l, cutoff=cutoff)
    IT_mps_invFTx_invFTy = fouriertransform(IT_mps_invFTx; sign=-1, tag="n", sitesdst=sites_l, cutoff=cutoff)

    # reconvert to quantics tt
    if reverse
        return TCI.reverse(TCI.TensorTrain(IT_mps_invFTx_invFTy))
    else
        return TCI.TensorTrain(IT_mps_invFTx_invFTy)
    end
end



function kinetic_evolution_2D(qtt_xy, exp_lap_momentum_mpo_IT, R; cutoff=1e-10)

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    sites_k = [Index(2, "Qubit,k=$k") for k in 1:R]
    sites_n = [Index(2, "Qubit,n=$n") for n in 1:R]
    sites_l = [Index(2, "Qubit,l=$l") for l in 1:R]
    sites_m2 = [Index(2, "Qubit,m2=$m") for m in 1:R]
    sites_k2 = [Index(2, "Qubit,k2=$k") for k in 1:R]
    sites_n2 = [Index(2, "Qubit,n2=$n") for n in 1:R]
    sites_l2 = [Index(2, "Qubit,l2=$l") for l in 1:R]
    sites_M = [Index(2, "Qubit,m=$m") for m in 1:2*R]

    # make interleaved collection of sites
    sites_mn = collect(Iterators.flatten(zip(sites_m, sites_n)))
    sites_mn2 = collect(Iterators.flatten(zip(sites_m2, sites_n2)))

    #transforming to ITensors
    IT_mps = MPS(TCI.TensorTrain(qtt_xy); sites=sites_mn)

    # Fourier transform for x, Fourier transform for y
    #IT_mps_FTx = (1/sqrt(2)^R) * fouriertransform(IT_mps; sign=1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    IT_mps_FTx = fouriertransform(IT_mps; sign=1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    #IT_mps_FTx_FTy = (1/sqrt(2)^R) * fouriertransform(IT_mps_FTx; sign=1, tag="n", sitesdst=sites_l, cutoff=cutoff)
    IT_mps_FTx_FTy = fouriertransform(IT_mps_FTx; sign=1, tag="n", sitesdst=sites_l, cutoff=cutoff)
    ft_psi_mps = TCI.reverse(TCI.TensorTrain(IT_mps_FTx_FTy))
    
    # apply kinetic operator in momentum space
    ft_psi_mps = apply_MPO_IT(ft_psi_mps, exp_lap_momentum_mpo_IT, sites_M, maxdim)
    
    
    #transforming to ITensors
    FT_IT_mps = MPS(TCI.TensorTrain(ft_psi_mps); sites=sites_mn2)

    # Fourier transform for x, Fourier transform for y
    #IT_mps_invFTx = (1/sqrt(2)^R) * fouriertransform(IT_mps; sign=-1, tag="m", sitesdst=sites_k, cutoff=cutoff)
    IT_mps_invFTx = fouriertransform(FT_IT_mps; sign=-1, tag="m", sitesdst=sites_k2, cutoff=cutoff)
    #IT_mps_invFTx_invFTy = (1/sqrt(2)^R) * fouriertransform(IT_mps_invFTx; sign=-1, tag="n", sitesdst=sites_l, cutoff=cutoff)
    IT_mps_invFTx_invFTy = fouriertransform(IT_mps_invFTx; sign=-1, tag="n", sitesdst=sites_l2, cutoff=cutoff)
    
    return TCI.reverse(TCI.TensorTrain(IT_mps_invFTx_invFTy))

    # reconvert to quantics tt
    #if reverse
    #    return TCI.reverse(TCI.TensorTrain(IT_mps_FTx_FTy))
    #else
    #    return TCI.TensorTrain(IT_mps_FTx_FTy)
    #end
end




## GP equation ##
function GP_Trotter_MPS_2D(psi0, pot, R, xmin, xmax, ymin, ymax, g, dt, Nsteps, m, tol, prec, maxdim, evaluate_wf=true)

    # constants
    M = 2^R
    vol_cell = (xmax-xmin)*(ymax-ymin)/M^2
    kcut = 2^8
    beta = 2
    #xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 50
    #cont_method = :TCI
    cont_method = :naive
    fourier_tol = 1e-8

    # save outputs
    bond_dimensions = zeros(Int64(Nsteps))
    wf_evolution = zeros(ComplexF64, (Int64(Nsteps), 2^prec, 2^prec))
    # zeros(ComplexF64, 2^prec, 2^prec)

    # construct initial quantics MPS directly
    #qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    #psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, xygrid; tolerance=tol)
    psi_mps = psi.tci
    println("psi_mps ", psi_mps)
    
    println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
    println("renormalising")
    psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
    println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))

    # construct relevant MPOs:
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:2*R]
    #qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=1e-10)
    #invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=1e-10)
    #exp_lap_momentum_mpo = exp_lap_Fourier_MPO_2D(xmin, xmax, ymin, ymax, R, dt, m, tol)
    exp_lap_momentum_mpo = exp_lap_Fourier_MPO_lowpass_2D(xmin, xmax, ymin, ymax, R, dt, m, tol, kcut, beta)
    exp_lap_momentum_mpo_IT = IT_MPO_conversion(exp_lap_momentum_mpo, sites_m)
    
    println("exp_lap_momentum_mpo ", exp_lap_momentum_mpo)
    exp_potential_mpo = exp_potential_MPO_2D(pot, dt, xmin, xmax, ymin, ymax, R, tol)
    println("exp_potential_mpo ", exp_potential_mpo)
    exp_potential_mpo_half = exp_potential_MPO_2D(pot, dt/2, xmin, xmax, ymin, ymax, R, tol)
    exp_potential_mpo_IT = IT_MPO_conversion(exp_potential_mpo, sites_m)
    exp_potential_mpo_half_IT = IT_MPO_conversion(exp_potential_mpo_half, sites_m)


    # do second-order Trotter evolution
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
    if g != 0
        psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt/2, tol)
    end

    for j in 1:Nsteps
        if j%1 == 0
            println("doing step j = ", j)
        end

	# kinetic evolution
        #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
        ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
        println("ft_psi_mps ", ft_psi_mps)
        ft_psi_mps = apply_MPO_IT(ft_psi_mps, exp_lap_momentum_mpo_IT, sites_m, maxdim)
        #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
        #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
        psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)
        
        # compact kinetic evolution
        #println("kinetic compact ", psi_mps)
        #psi_mps = kinetic_evolution_2D(psi_mps, exp_lap_momentum_mpo_IT, R; cutoff=1e-10)
        
        println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
        println("renormalising")
        psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
        println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))

        # potential evolution
        #psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
        psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_IT, sites_m, maxdim)
        if g != 0
            psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt, tol)
        end
        println("apply_f_tt_2D ", psi_mps)

        # evaluate wavefunction if desired
        #psi_mps.compress(tol)
        #print("bond dimension ", psi_mps.bond)
        bond_dimensions[j] = get_max_rank(psi_mps)

        if evaluate_wf
            #psi_ev = MPS_to_TT(psi_mps)
            #wf_evolution[j, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
            wf_evolution[j, :, :] = evaluate_2D(psi_mps, xygrid, R, prec)
        end


    end
    #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
    #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol)
    #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)

    #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
    ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
    #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
    ft_psi_mps = apply_MPO_IT(ft_psi_mps, exp_lap_momentum_mpo_IT, sites_m, maxdim)
    #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
    psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)
    
    println("norm before = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))
    println("renormalising")
    psi_mps = normalise_2D(psi_mps, R, xmin, xmax, ymin, ymax)
    println("norm after = ", fidelity_ITensor_2D(psi_mps, psi_mps, R, xmin, xmax, ymin, ymax))

    # potential evolution
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
    psi_mps = apply_MPO_IT(psi_mps, exp_potential_mpo_half_IT, sites_m, maxdim)
    if g != 0
        psi_mps = apply_f_tt_2D(psi_mps, xmin, xmax, ymin, ymax, R, g, dt/2, tol)
    end

    # maybe return qgrid?
    return wf_evolution, bond_dimensions
end



############
## old stuff
############



function Schrödinger_Trotter_MPS_1D(psi0, pot, R, xmin, xmax, dt, Nsteps, m, tol, prec, maxdim, evaluate_wf=true)

    # constants
    M = 2^R
    vol_cell = (xmax-xmin)/M
    xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 50
    cont_method = :TCI

    # save outputs
    bond_dimensions = zeros(Nsteps)
    wf_evolution = Complex.(zeros(Nsteps, 2^prec))

    # construct initial quantics MPS directly
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    psi_mps = psi.tci

    # construct relevant MPOs:
    fourier_tol = 1e-10
    qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=1e-10)
    invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=1e-10)
    exp_lap_momentum_mpo = exp_lap_Fourier_MPO(xmin, xmax, R, dt, m, tol)
    exp_potential_mpo = exp_potential_MPO(pot, dt, xmin, xmax, R, tol)
    exp_potential_mpo_half = exp_potential_MPO(pot, dt/2, xmin, xmax, R, tol)


    # MPOs for width
    #if calculate_width:
    #    def pos_operator(x):
    #        return x
    #    def pos_operator_sq(x):
    #        return x**2
    #    pos_mpo = potential_MPO(pos_operator, xmin, xmax, R, maxdim, operator_tol, numsweeps)
    #    pos_sq_mpo = potential_MPO(pos_operator_sq, xmin, xmax, R, maxdim, operator_tol, numsweeps)


    # do second-order Trotter evolution
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)
    for j in 1:Nsteps
        if j%5 == 0
            println("doing step j = ", j)
        end

        ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
        ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol)
        psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
        psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=cont_method, tolerance=tol)

        # evaluate wavefunction if desired
        #psi_mps.compress(tol)
        #print("bond dimension ", psi_mps.bond)
        bond_dimensions[j] = get_max_rank(psi_mps)

        if evaluate_wf
            #psi_ev = MPS_to_TT(psi_mps)
            wf_evolution[j, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
        end


    end
    ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
    ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol)
    psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)

    # maybe return qgrid?
    return wf_evolution, bond_dimensions, xs
end


## Schrödinger equation ##
function Schrödinger_Trotter_MPS_2D(psi0, pot, R, xmin, xmax, ymin, ymax, dt, Nsteps, m, tol, prec, maxdim, evaluate_wf=true)

    # constants
    M = 2^R
    vol_cell = (xmax-xmin)*(ymax-ymin)/M^2
    #xs = LinRange(xmin, xmax, 2^prec)
    #maxdim = 50
    #cont_method = :TCI
    cont_method = :naive

    # save outputs
    bond_dimensions = zeros(Int64(Nsteps))
    wf_evolution = zeros(ComplexF64, (Int64(Nsteps), 2^prec, 2^prec))
    # zeros(ComplexF64, 2^prec, 2^prec)

    # construct initial quantics MPS directly
    #qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    #psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, qgrid, tolerance=tol)
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    psi, _, _ = quanticscrossinterpolate(ComplexF64, psi0, xygrid; tolerance=1e-10)
    psi_mps = psi.tci
    println("psi_mps ", psi_mps)

    # construct relevant MPOs:
    fourier_tol = 1e-10
    #qftmpo = quanticsfouriermpo(R; sign=-1.0, normalize=true, tolerance=1e-10)
    #invqftmpo = quanticsfouriermpo(R; sign=1.0, normalize=true, tolerance=1e-10)
    exp_lap_momentum_mpo = exp_lap_Fourier_MPO_2D(xmin, xmax, ymin, ymax, R, dt, m, tol)
    println("exp_lap_momentum_mpo ", exp_lap_momentum_mpo)
    exp_potential_mpo = exp_potential_MPO_2D(pot, dt, xmin, xmax, ymin, ymax, R, tol)
    println("exp_potential_mpo ", exp_potential_mpo)
    exp_potential_mpo_half = exp_potential_MPO_2D(pot, dt/2, xmin, xmax, ymin, ymax, R, tol)


    # MPOs for width
    #if calculate_width:
    #    def pos_operator(x):
    #        return x
    #    def pos_operator_sq(x):
    #        return x**2
    #    pos_mpo = potential_MPO(pos_operator, xmin, xmax, R, maxdim, operator_tol, numsweeps)
    #    pos_sq_mpo = potential_MPO(pos_operator_sq, xmin, xmax, R, maxdim, operator_tol, numsweeps)


    # do second-order Trotter evolution
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)
    for j in 1:Nsteps
        if j%1 == 0
            println("doing step j = ", j)
        end

        #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
        ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
        println("ft_psi_mps ", ft_psi_mps)
        ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
        #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
        psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)

        # potential evolution
        psi_mps = TCI.contract(exp_potential_mpo, psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)

        # evaluate wavefunction if desired
        #psi_mps.compress(tol)
        #print("bond dimension ", psi_mps.bond)
        bond_dimensions[j] = get_max_rank(psi_mps)

        if evaluate_wf
            #psi_ev = MPS_to_TT(psi_mps)
            #wf_evolution[j, :] = evaluate_wavefunction(psi_mps, qgrid, prec, xmax)
            wf_evolution[j, :, :] = evaluate_2D(psi_mps, xygrid, R, prec)
        end


    end
    #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
    #ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol)
    #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
    #psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol)

    #ft_psi_mps = TCI.reverse(TCI.contract(qftmpo, psi_mps; algorithm=cont_method, tolerance=tol))
    ft_psi_mps = Fourier_transform_2D(psi_mps, R, cutoff=fourier_tol, reverse=true)
    ft_psi_mps = TCI.contract(exp_lap_momentum_mpo, ft_psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)
    #psi_mps = TCI.reverse(TCI.contract(invqftmpo, ft_psi_mps; algorithm=cont_method, tolerance=tol))
    psi_mps = inv_Fourier_transform_2D(ft_psi_mps, R, cutoff=fourier_tol, reverse=true)

    # potential evolution
    psi_mps = TCI.contract(exp_potential_mpo_half, psi_mps; algorithm=cont_method, tolerance=tol, maxbonddim=maxdim)

    # maybe return qgrid?
    return wf_evolution, bond_dimensions
end
