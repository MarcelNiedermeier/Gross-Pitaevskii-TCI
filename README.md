Code to solve the Gross-Pitaevskii equation in one and two dimensions via a mixed-spectral second-order Trotterisation algorithm, based on matrix product states and tensor cross interpolation. 

There are two identical directories in the working_example folder for one and two dimensions. In the following, I will describe the structure of the 1D folder; the 2D folder works in the same fashion.

The packages with which this code has been tested and executed are the following:

'''
Pkg.add([
    PackageSpec(name="BenchmarkTools", version="1.6.0"),
    PackageSpec(name="DelimitedFiles", version="1.9.1"),
    PackageSpec(name="FFTW", version="1.8.1"),
    PackageSpec(name="ITensorMPS", version="0.3.6"),
    PackageSpec(name="ITensors", version="0.7.13"),
    PackageSpec(name="JLD2", version="0.5.13"),
    PackageSpec(name="LaTeXStrings", version="1.4.0"),
    PackageSpec(name="Plots", version="1.40.13"),
    PackageSpec(name="Quantics", version="0.4.5"),
    PackageSpec(name="QuanticsGrids", version="0.3.3"),
    PackageSpec(name="QuanticsTCI", version="0.7.0"),
    PackageSpec(name="TCIITensorConversion", version="0.2.0"),
    PackageSpec(name="TensorCrossInterpolation", version="0.9.14"),
    PackageSpec(name="TensorOperations", version="5.1.4"),
    PackageSpec(name="LinearAlgebra", version="1.11.0"),
])

'''
The Julia version is Julia 1.11.1.


General (helper) functions are contained in the 'utilities.jl' file. 

A given simulation of the Gross-Pitaevskii equation can then be executed by running the 'GP_1D.jl' file.
