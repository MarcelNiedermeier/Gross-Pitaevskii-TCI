using Pkg

# Start a new environment in the current folder
Pkg.activate(".")
Pkg.instantiate()

# Add each package with its specific version
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

