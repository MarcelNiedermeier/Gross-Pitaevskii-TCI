

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

function get_max_rank(tt)
    ranks = []
    for i in 1:length(tt)
        #println("check size: ", size(tt[i])[3])
        push!(ranks, size(tt[i])[3])
    end
    return maximum(ranks)
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

function evaluate_reduced_2D(qtt_xy, xygrid, R, prec, xmin, xmax, ymin, ymax)
    """ Reconstruct the image of a quantics crossinterpolated function in 2D.
    Version for TT object. """
    xminindex = QG.origcoord_to_grididx(xygrid, (xmin,0.))
    xmaxindex = QG.origcoord_to_grididx(xygrid, (xmax,0.))
    yminindex = QG.origcoord_to_grididx(xygrid, (0.,ymin))
    ymaxindex = QG.origcoord_to_grididx(xygrid, (0.,ymax))
    xindexgrid = round.(Int, LinRange(xminindex[1], xmaxindex[1], 2^prec))
    yindexgrid = round.(Int, LinRange(yminindex[2], ymaxindex[2], 2^prec))
    arr = zeros(ComplexF64, 2^prec, 2^prec)
    for i in 1:2^prec
        for j in 1:2^prec
            arr[i, j] = qtt_xy(QG.grididx_to_quantics(xygrid, (xindexgrid[i], yindexgrid[j])))
        end
    end
    return arr
end




function x_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function xpos(x, y)
        return x
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, xpos, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function x_squared_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function xpos_squared(x, y)
        return x^2 
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, xpos_squared, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function y_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function ypos(x, y)
        return y
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, ypos, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function y_squared_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function ypos_squared(x, y)
        return y^2 
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, ypos_squared, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function r_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function r(x, y)
        return sqrt(x^2 + y^2)
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, r, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end


function r_squared_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    function r_squared(x, y)
        return x^2 + y^2
    end
    xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)
    ci, ranks, errors = quanticscrossinterpolate(ComplexF64, r_squared, xygrid, tolerance=tol)
    return TT_to_MPO(ci.tci)
end



function expectation_value_ITensor_2D(psi, op, R, xmin, xmax, ymin, ymax)
    tol = 1e-8
    dx = (xmax-xmin)/2^R
    dy = (ymax-ymin)/2^R

    op_psi = TCI.contract(op, psi; algorithm=:naive, tolerance=tol)

    # get sites for ITensors
    sites_m = [Index(2, "Qubit,m=$m") for m in 1:2*R]

    #transforming to ITensors
    IT_psi = MPS(psi; sites=sites_m)
    IT_op_psi = MPS(op_psi; sites=sites_m)

    return real(ITensors.inner(IT_psi, IT_op_psi)*dx*dy)
    #return abs(ITensors.inner(IT_psi1, IT_psi2)*dx)^2
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
R = 20
dt = 0.01
xmin, xmax = -100., 100.
ymin, ymax = -100., 100.
#xmin, xmax = 0., 2*pi
#ymin, ymax = 0., 2*pi
maxdim = 10
prec = 10
tol = 1e-8
#tols = [0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
Nsteps = 116
omega1 = 0.01
#omega2 = 10000.0
omega2 = 10
omega3 = 100000.0
#omega3 = 1.618033988749895
#omega3 = 1.5
g = 1 # 0, 5
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
A3s = [40.0]



# what do you want to compute?
compute_pos = false
compute_width = false
compute_heatmap_full = true
compute_heatmap_reduced = true
compare_exact = false # check fidelity MPS vs analytical solution
compute_bond_dim = false

# choose at which points in time you would like to calculate the heatmaps!
#heatmap_indices = [1, 111, 221, 331]
#heatmap_indices = [1, 25, 50, 75, 100, 125, 150]#, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
#heatmap_indices = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]#, 120, 130, 140, 150]
heatmap_indices = [30, 80]
#heatmap_indices = [1, 50, 100, 150]
#xreds = [10., 1.]#, 5.]
#yreds = [10., 1.]#, 5.]
#xreds = [40., 10.]#, 5.]
#yreds = [40., 10.]#, 5.]
#xreds = [0.25]#, 5.]
#yreds = [0.25]#, 5.]
xreds = [2.5]#, 5.]
yreds = [2.5]#, 5.]
x0_red = 0.
y0_red = 0.
wf_evolution = zeros(ComplexF64, (Int64(length(heatmap_indices)), 2^prec, 2^prec))
wf_evolution_red = zeros(ComplexF64, (Int64(length(xreds)), Int64(length(heatmap_indices)), 2^prec, 2^prec))

xygrid = QG.DiscretizedGrid{2}(R, (xmin, ymin), (xmax, ymax); unfoldingscheme=:interleaved)


if compute_pos
    xpos = zeros(Nsteps)
    ypos = zeros(Nsteps)
    xpos_mpo = x_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    ypos_mpo = y_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    println("pos_mpo ", xpos_mpo)
end


if compute_width
    xwidths = zeros(Nsteps)
    ywidths = zeros(Nsteps)
    xpos_mpo = x_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    xpos_squared_mpo = x_squared_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    ypos_mpo = y_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    ypos_squared_mpo = y_squared_MPO_2D(xmin, xmax, ymin, ymax, R, tol)
    println("pos_mpo ", xpos_mpo)
    println("pos_squared_mpo ", xpos_squared_mpo)
end

if compare_exact
    fids = zeros(Nsteps)
end

if compute_bond_dim
    bond_dims = zeros(Nsteps)
end


# load MPS wave functions
cd("Data_Gaussian_moving_k5_quadratic_potential_2D_sine_potential_modulation_x_2D_sine_potential_modulation_y_2D_sine_potential_modulation_twist1_2D_sine_potential_modulation_twist2_2D_20_g_5_xy_100.0_Nsteps_10000_maxdim_50_tol_1.0e-8_dt_0.01/")
#cd("Data_Gaussian_moving_quadratic_potential_2D_sine_potential_modulation_x_2D_sine_potential_modulation_y_2D_sine_potential_modulation_twist1_2D_sine_potential_modulation_twist2_2D_20_g_5_xy_100.0_Nsteps_10000_maxdim_50_tol_1.0e-8_dt_0.01")
#cd("Data_quadratic_potential_2D_sine_potential_modulation_2D_20_g_5_xy_100.0_Nsteps_10000_maxdim_50_tol_1.0e-8_dt_0.01")
#cd("Data_quadratic_potential_aniso_anharm_0.01_0.015_0.012_R_20_g_5_xy_100_Nsteps_5000_maxdim_50_tol_1e-8_dt_0.01/")
#cd("Data_quadratic_potential_iso_0.01_R_20_g_5_xy_100_Nsteps_5000_maxdim_50_tol_1e-8_dt_0.01/")
#cd("Data_quadratic_potential_aniso_2D_sine_potential_modulation_2D_20_g_5_xy_100.0_Nsteps_10000_maxdim_50_tol_1.0e-8_dt_0.01")
#cd("Data_quadratic_potential_2D_sine_potential_modulation_x_2D_sine_potential_modulation_y_2D_sine_potential_modulation_twist1_2D_sine_potential_modulation_twist2_2D_20_g_5_xy_100.0_Nsteps_10000_maxdim_50_tol_1.0e-8_dt_0.01")
println("loading files")
wfs_mps = []
files = filter(f -> endswith(f, ".jld2"), readdir("."))
sorted_files = sort(files, by = x -> parse(Int, match(r"\d+", x).match))
for i in 1:length(files)
	push!(wfs_mps, load_object(sorted_files[i]))
end
println("files loaded!")



if compute_pos
    println("computing position")
        for i in 1:Nsteps
	    if i%1 == 0
	        println("step i = $i")
	    end
	    xpos[i] = expectation_value_ITensor_2D(wfs_mps[i], xpos_mpo, R, xmin, xmax, ymin, ymax)
	    ypos[i] = expectation_value_ITensor_2D(wfs_mps[i], ypos_mpo, R, xmin, xmax, ymin, ymax)
	    println("xpos = ", xpos)
            println("ypos = ", ypos)
        end
    writedlm("Data_reconstructed/xpos.txt", xpos, ",")
    writedlm("Data_reconstructed/ypos.txt", ypos, ",")
end


if compute_width
    println("computing width")
    for i in 1:Nsteps
        if i%1 == 0
            println("step i = $i")
        end
        xpos = expectation_value_ITensor_2D(wfs_mps[i], xpos_mpo, R, xmin, xmax, ymin, ymax)
        xpos_squared = expectation_value_ITensor_2D(wfs_mps[i], xpos_squared_mpo, R, xmin, xmax, ymin, ymax)
        ypos = expectation_value_ITensor_2D(wfs_mps[i], ypos_mpo, R, xmin, xmax, ymin, ymax)
        ypos_squared = expectation_value_ITensor_2D(wfs_mps[i], ypos_squared_mpo, R, xmin, xmax, ymin, ymax)
        x_width = sqrt(xpos_squared - xpos^2)
        y_width = sqrt(ypos_squared - ypos^2)
        println("x_width = ", x_width)
        println("y_width = ", y_width)
        xwidths[i] = x_width
        ywidths[i] = y_width
    end
    writedlm("Data_reconstructed/xwidths.txt", xwidths, ",")
    writedlm("Data_reconstructed/ywidths.txt", ywidths, ",")
end

if compare_exact
    println("comparing to analytical solution")
    local t = 0
    for i in 1:Nsteps
        if i%1 == 0
            println("step i = $i")
        end
        #println("t = ", t)
        function sine_exact_2D(x, y)
            println("t = ", t)
	    return sin(x)*sin(y)*exp(-2*im*t)
	end
	println("func = ", sine_exact_2D(1, 1))


        t += 10*dt
    end
end



if compare_exact
    println("comparing to analytical solution")
    local t = 0
    for i in 1:Nsteps
        if i%1 == 0
            println("step i = $i")
        end
        
        function sine_exact_2D(x, y)
	    return sin(x)*sin(y)*exp(-2*im*t)
	end

        sin_ex, _, _ = quanticscrossinterpolate(ComplexF64, sine_exact_2D, xygrid; tolerance=tol)
        sin_ex_mps = sin_ex.tci
        println("norm before = ", fidelity_ITensor_2D(sin_ex_mps, sin_ex_mps, R, xmin, xmax, ymin, ymax))
        println("renormalising")
        sin_ex_mps = normalise_2D(sin_ex_mps, R, xmin, xmax, ymin, ymax)
        fid = fidelity_ITensor_2D(sin_ex_mps, wfs_mps[i], R, xmin, xmax, ymin, ymax)
        println("fidelity with analytical: ", fid)
        fids[i] = fid
        t += 10*dt
    end
    writedlm("Data_reconstructed/fidelities_analytical.txt", fids, ",")
end



if compute_heatmap_full	
    println("computing heatmap full")
    for k in 1:length(heatmap_indices)
        println("calculating heatmap $(k)")
        #wf_evolution[k, :, :] = evaluate_2D(wfs_mps[k], xygrid, R, prec)
        wf = evaluate_2D(wfs_mps[k], xygrid, R, prec)
        wf_evolution[k, :, :] = wf
    	writedlm("Data_reconstructed/wf_2D_$(k).txt", wf, ",")
    end
end

if compute_heatmap_reduced
    for i in 1:length(xreds)
        println("computing heatmap reduced")
        for k in 1:length(heatmap_indices)
            println("calculating heatmap reduced $(heatmap_indices[k])")
            wf_red = evaluate_reduced_2D(wfs_mps[heatmap_indices[k]], xygrid, R, prec, x0_red-xreds[i], x0_red+xreds[i], y0_red-yreds[i], y0_red+yreds[i])
            wf_evolution_red[i, k, :, :] = wf_red
    	    writedlm("Data_reconstructed/wf_red_2D_$(x0_red)_$(y0_red)_$(xreds[i])_$(k).txt", wf_red, ",")
    	end
    end
end


if compute_bond_dim
    println("computing maximum bond dimension")
    for i in 1:Nsteps
        if i%1 == 0
             println("step i = $i")
        end
        bdim = get_max_rank(wfs_mps[i])
        println("bond dim = ", bdim)
        bond_dims[i] = bdim
        writedlm("Data_reconstructed/bond_dims.txt", bond_dims, ",")
    end
end


############
# quick plot
############

if compute_pos
    p1 = plot(1:Nsteps, xpos[:], label="x", legend=:topleft)
    plot!(1:Nsteps, ypos[:], label="y")
    savefig(p1, "Plots/pos_exp_value_2D_test.png")
end


if compute_width
    p1 = plot(1:Nsteps, xwidths[:], label="x width", legend=:topleft)
    plot!(1:Nsteps, ywidths[:], label="y width")
    savefig(p1, "Plots/widths_2D_test.png")
end


if compare_exact
    p1 = plot(1:Nsteps, fids[:], label="fidelities analytical", legend=:topleft)
    savefig(p1, "Plots/fidelities_analytical_2D_test.png")
end


if compute_heatmap_full
    for k in 1:length(heatmap_indices)
        heatmap(abs.(wf_evolution[k, :, :]).^2)
        savefig("Plots/wf_2D_$(heatmap_indices[k]).png")
    end
end


if compute_heatmap_reduced
    for i in 1:length(xreds)
        for k in 1:length(heatmap_indices)
            heatmap(abs.(wf_evolution_red[i, k, :, :]).^2)
            savefig("Plots/wf_red_2D_$(x0_red)_$(y0_red)_$(xreds[i])_$(heatmap_indices[k]).png")
        end
    end
end

if compute_bond_dim
    p1 = plot(1:Nsteps, bond_dims[:], label="bond dim", legend=:topleft)
    savefig(p1, "Plots/bond_dims_2D_test.png")
end


hjfkjfdkj




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

    cd("..")
end


