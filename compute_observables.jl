

############################################
## observable calculation over multiple runs
############################################

using LaTeXStrings
using Plots
gr()

import QuanticsGrids as QG
using QuanticsTCI: quanticscrossinterpolate, quanticsfouriermpo, integral
import TensorCrossInterpolation as TCI
using TensorOperations

import TCIITensorConversion
using ITensors
using ITensorMPS
import Quantics: fouriertransform, Quantics

using JLD2
using BenchmarkTools
using DelimitedFiles

using Statistics


###########
# functions
###########

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

function comb_even_odd_MPO(xmin, xmax, R, tol)
    beta = 5
    function comb_even_odd(x)
        return tanh(beta*cos(omega2*x))
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, tolerance=1e-6, nrandominitpivot=1000)
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, maxdim=100, nrandominitpivot=1000)
    return TT_to_MPO(ci.tci)
end


function cos_comb_even_odd_mps(xmin, xmax, R, tol)
    beta = 5
    function comb_even_odd(x)
        return cos(omega2*x)
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, tolerance=1e-6, nrandominitpivot=1000)
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, maxdim=100, nrandominitpivot=1000)
    return ci.tci
end


function comb_even_odd_mps(xmin, xmax, R, omega, tol)
    beta = 5
    function comb_even_odd(x)
        return tanh(beta*cos(omega*x))
    end
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, tolerance=1e-6, nrandominitpivot=1000)
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, comb_even_odd, qgrid, maxdim=100, nrandominitpivot=1000)
    return ci.tci
end


#function apply_f_tt(psi, R, xmin, xmax, g, dt, tol, maxdim)
#    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
#    function exp_nonlinear(x)
#        ind = QG.origcoord_to_grididx(qgrid, x)
#        wf = TCI.evaluate(psi, QG.grididx_to_quantics(qgrid, ind))
#        return exp(-1.0im*g*abs(wf)^2 * dt)*wf
#    end
#    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, exp_nonlinear, qgrid, tolerance=tol, maxbonddim=maxdim)
#    return ci.tci
#end


function psi_squared_mps(psi_mps, xmin, xmax, R, tol, maxdim)
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    function psi_squared(x)
        ind = QG.origcoord_to_grididx(qgrid, x)
        wf = TCI.evaluate(psi_mps, QG.grididx_to_quantics(qgrid, ind))
        return abs(wf)
    end
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, psi_squared, qgrid, tolerance=tol, maxbonddim=maxdim)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, psi_squared, qgrid, tolerance=1e-6, maxbonddim=maxdim)
    return ci.tci
end

function psi_squared_comb_ci(psi_mps, xmin, xmax, R, omega, tol, maxdim)
    qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)
    beta = 5
    function psi_squared(x)
        ind = QG.origcoord_to_grididx(qgrid, x)
        wf = TCI.evaluate(psi_mps, QG.grididx_to_quantics(qgrid, ind))
        return tanh(beta*cos(omega*x))*abs(wf)^2
    end
    #ci, ranks, errors = quanticscrossinterpolate(ComplexF64, psi_squared, qgrid, tolerance=tol, maxbonddim=maxdim)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, psi_squared, qgrid, tolerance=1e-8, maxbonddim=maxdim)
    return ci
end


function expectation_value_ITensor(psi, op, R, xmin, xmax)
    tol = 1e-8
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


function apply_MPO_IT(psi_mps, mpo_IT, sites, maxdim, nsweeps=2)
    #sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
    #psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi.tci); sites=sites_m)
    psi_mps_IT = ITensorMPS.MPS(TCI.TensorTrain(psi_mps); sites=sites)
    psi_mpo_mps_IT = ITensors.contract(mpo_IT, psi_mps_IT; method="fit", nsweeps=nsweeps, maxdim=maxdim)
    return TCI.TensorTrain(psi_mpo_mps_IT)
end


##############
# computations
##############

# constants
R = 30
dt = 0.01
xmin = -500.
xmax = 500.
maxdim = 10
prec = 12
tol = 1e-6
#tols = [0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
Nsteps = 500
omega1 = 0.01
#omega2 = 10000.0
omega2 = 10
omega3 = 100000.0
#omega3 = 1.618033988749895
#omega3 = 1.5
g = 5 # 0, 5
#gs = [0, 1, 5]
#A2s = [0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#A2s = [0.01, 1.0, 2.0, 3.0, 4.0]
#A2s = [0.01, 1.0, 2.0, 3.0]
#A2s = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]#, 6.0, 7.0, 8.0, 9.0]
#A2s = [0.1, 1.0, 2.0, 3.0, 4.0]
#A2s = [10.0]
A2 = 5.0
#A3s = [0.2, 0.4]#, 0.6, 0.8]#, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#A3s = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
#A3s = [0.2, 2.0]
A3s = [20.0]
#xreds = [10., 1., 0.2, 2*1e-5]#, 2., 0.1]
#A3 = 40.0
#xreds = [1, 0.1, 2*1e-2, 1e-3, 1e-4, 2*1e-5]
#xreds = [10., 5., 1e-4, 4*1e-5]
xreds = [1e-4]
x0_red = 5.

# what do you want to compute?
compute_width = false
compute_imbalance = false
compute_heatmap_full = true
compute_heatmap_reduced = true

qgrid = QG.DiscretizedGrid{1}(R, xmin, xmax; includeendpoint=true)

if compute_width
    widths = zeros(length(A3s), Nsteps)
    pos_mpo = pos_MPO(xmin, xmax, R, tol)
    pos_squared_mpo = pos_squared_MPO(xmin, xmax, R, tol)
    println("pos_mpo ", pos_mpo)
    println("pos_squared_mpo ", pos_squared_mpo)
end

if compute_imbalance
    imbalances = zeros(length(A3s), Nsteps)
    av_imbalances = zeros(length(A3s), Nsteps)
    #comb_mpo = comb_even_odd_MPO(xmin, xmax, R, tol)
    comb_mps = comb_even_odd_mps(xmin, xmax, R, omega2, tol)
    #comb_mps = cos_comb_even_odd_mps(xmin, xmax, R, tol)
    println("comb_mps ", comb_mps)
end

# loop through directories and compute desired quantities
for k in 1:length(A3s)
    println("doing A3 = ", A3s[k])

    # load MPS wave functions
    # wf_evolution_sine_potential2_sine_potential3_MPS_1D_R_30_Nsteps_5000_dt_0.01_g_0_maxdim_12_x_500.0_O2_1.0_O3_1618.033988749895_A2_3.0_A3_2.0
    cd("wf_evolution_sine_potential2_sine_potential3_MPS_1D_R_30_Nsteps_5000_dt_0.01_g_$(g)_maxdim_14_x_500.0_O1_0.01_O2_$(omega2)_O3_$(omega3)_A2_$(A2)_A3_$(A3s[k])")
    #cd("wf_evolution_quadratic_potential_sine_potential2_MPS_1D_R_$(R)_Nsteps_5000_dt_$(dt)_g_$(g)_tol_$(tols[k])_x_100.0_O1_$(omega1)_O2_$(omega2)_A2_$(A2)")
    #cd("wf_evolution_quadratic_potential_sine_potential2_MPS_1D_R_$(R)_Nsteps_5000_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_100.0_O1_$(omega1)_O2_$(omega2)_A2_$(A2s[k])")
    #cd("wf_evolution_quadratic_potential_sine_potential2_sine_potential3_MPS_1D_R_$(R)_Nsteps_5000_dt_$(dt)_g_$(g)_maxdim_$(maxdim)_x_100.0_O1_$(omega1)_O2_$(omega2)_O3_$(omega3)_A2_$(A2s[k])_A3_$(A3)")
    println("loading files")
    wfs_mps = []
    files = filter(f -> endswith(f, ".jld2"), readdir("."))
    sorted_files = sort(files, by = x -> parse(Int, match(r"\d+", x).match))
    for i in 1:length(files)
        push!(wfs_mps, load_object(sorted_files[i]))
    end
    println("files loaded!")

    if compute_heatmap_full
        println("computing heatmap full")
        wf_evolution = Complex.(zeros(Nsteps, 2^prec))
        for i in 1:Nsteps
            if i%50 == 0
                println("step i = $i")
            end
            wf_evolution[i, :] = evaluate_wavefunction(wfs_mps[i], qgrid, prec, xmax)
        end
        writedlm("Data_reconstructed/wf_evolution.txt", wf_evolution, ",")
    end

    if compute_heatmap_reduced
        println("computing heatmap reduced")
        wf_red = []
        for i in 1:length(xreds)
            println("xred = ", xreds[i])
            wf_evolution_red = Complex.(zeros(Nsteps, 2^prec))
            for j in 1:Nsteps
                if j%50 == 0
                    println("step j = $j")
                end
                wf_evolution_red[j, :] = evaluate_wavefunction_reduced(wfs_mps[j], qgrid, prec, x0_red-xreds[i], x0_red+xreds[i])
            end
            push!(wf_red, wf_evolution_red)
        end
        for i in 1:length(xreds)
            writedlm("Data_reconstructed/wf_evolution_red_$(x0_red)_$(xreds[i]).txt", wf_red[i], ",")
        end
    end

    if compute_width
        println("computing width")
        for i in 1:Nsteps
            if i%50 == 0
                println("step i = $i")
            end
            pos = expectation_value_ITensor(wfs_mps[i], pos_mpo, R, xmin, xmax)
            pos_squared = expectation_value_ITensor(wfs_mps[i], pos_squared_mpo, R, xmin, xmax)
            widths[k, i] = sqrt(pos_squared - pos^2)
        end
        writedlm("Data_reconstructed/widths.txt", widths, ",")
    end

    if compute_imbalance
        println("computing imbalance")
        sites_m = [Index(2, "Qubit,m=$m") for m in 1:R]
        comb_mps_IT = MPS(comb_mps; sites=sites_m)
        for i in 1:Nsteps
            if i%10 == 0
                println("step i = $i")
            end

            # normalise
            psi_mps = normalise(wfs_mps[i], R, xmin, xmax)

            # compute with overlaps
            #dx = (xmax-xmin)/2^R
            #psi_squared = psi_squared_mps(psi_mps, xmin, xmax, R, tol, maxdim)
            #psi_squared_IT = MPS(psi_squared; sites=sites_m)
            #imbalance = ITensors.inner(psi_squared_IT, comb_mps_IT)*dx
            #println("imbalance: ", imbalance)

            # try with quantics integration
            weighted_ci = psi_squared_comb_ci(psi_mps, xmin, xmax, R, omega2, tol, maxdim)
            imbalance = integral(weighted_ci)
            println("imbalance: ", imbalance)

            imbalances[k, i] = imbalance
            av_imbalances[k, i] = mean(imbalances[k, 1:i])
        end
        writedlm("Data_reconstructed/imbalances.txt", imbalances[k, :], ",")
        writedlm("Data_reconstructed/av_imbalances.txt", av_imbalances[k, :], ",")
    end

    cd("..")
end

############
# quick plot
############

if compute_width
    p1 = plot(1:Nsteps, widths[1, :], label="A3 = $(A3s[1])", legend=:topleft)
    for k in 2:length(A3s)
        plot!(1:Nsteps, widths[k, :], label="A3 = $(A3s[k])")
    end
    savefig(p1, "AAH_even_comb_width_test_g_$(g).png")
end

if compute_imbalance
    p1 = plot(1:Nsteps, av_imbalances[1, :], label="pot height = $(A3s[1])", legend=:topleft)
    for k in 2:length(A3s)
        plot!(1:Nsteps, av_imbalances[k, :], label="pot height = $(A3s[k])")
    end
    xlabel!("time")
    ylabel!("av imbalance")
    savefig(p1, "AAH_incomm_even_comb_av_imbalance_test_g_$(g)_A2_$(A2).png")

    p2 = plot(1:Nsteps, imbalances[1, :], label="pot height = $(A3s[1])", legend=:topleft)
    for k in 2:length(A3s)
        plot!(1:Nsteps, imbalances[k, :], label="pot height = $(A3s[k])")
    end
    xlabel!("time")
    ylabel!("imbalance")
    savefig(p2, "AAH_incomm_even_comb_imbalance_test_g_$(g)_A2_$(A2).png")
end

# plot
#h2 = heatmap(abs.(wf_evolution).^2)
#savefig(h2, "Plots/GP_evolution_test.png")

#for i in 1:length(xreds)
#    h = heatmap(abs.(wf_red[i]).^2)
#    savefig(h, "Plots/GP_evolution_red_$(xreds[i])_test.png")
#end
