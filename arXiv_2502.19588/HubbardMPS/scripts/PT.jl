##################
# INITIALISATION #
##################

using DrWatson
@quickactivate "HubbardMPS"

# using BLISBLAS
using MPSKit
using KrylovKit
using TensorKit

include(projectdir("src", "HubbardFunctions.jl"))
import .HubbardFunctions as hf

# Extract name of the current file. Will be used as code name for the simulation.
name_jl = splitpath(Base.source_path())
name = first(split(last(name_jl),"."))


#################
# DEFINE SYSTEM #
#################

# Provide Schmidt cut value as FIRST ARGUMENT (s=4.0 is already considered good)
s = parse(Float64, ARGS[1])

include(projectdir("data", "params", "PT", "params.jl"))

P = 4;
Q = 3;
bond_dim = 40;

model = hf.MB_Sim(t[:,1:12], U[:,1:12], zeros(6,6), P, Q, s, bond_dim; code = name*"$s");


########################
# COMPUTE GROUNDSTATES #
########################

Force = false
code = get(model.kwargs, :code, "")

if !isdir(datadir("jld2",code)) || Force
    if !isdir(datadir("jld2"))
        mkdir(datadir("jld2"))
    end
    println("Computing ground state...")
    dictionary = hf.compute_groundstate(model; tol=1e-6);
    ψ₀ = dictionary["groundstate"];
    hf.save_state(ψ₀, datadir("jld2"), code)
    H = dictionary["ham"];
else
    ψ₀ = hf.load_state(datadir("jld2", code))
    H = hf.hamiltonian(model);
end
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
println("Groundstate energy: $E")
println("Bond dimension: $(hf.dim_state(ψ₀))")

println("Number of electrons: ", hf.density_state(ψ₀,model.P,model.Q,false))


#######################
# COMPUTE EXCITATIONS #
#######################

if length(ARGS) > 1
    resolution = 5;
    momenta = range(0, π, resolution);
    nums = 1;

    solver = Arnoldi(;krylovdim=25,tol=1e-5,eager=true)
    charges = [1,1/2,1]
    sector1 = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2]) ⊠ U1Irrep(charges[3]*Q)
    sector2 = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2]) ⊠ U1Irrep(-charges[3]*Q)
    # Provide as SECOND ARGUMENT "hole" or "particle" to compute the corresponding excitation
    if ARGS[2] == "hole"
        println("Computing hole excitation...")
        E1,_ = excitations(H, QuasiparticleAnsatz(solver), momenta, ψ₀; num=nums, sector=sector1)
        println("hole excitation: ", E1)
    elseif ARGS[2] == "particle"
        println("Computing particle excitation...")
        E2,_ = excitations(H, QuasiparticleAnsatz(solver), momenta, ψ₀; num=nums, sector=sector2)
        println("particle excitation: ", E2)
    else
        error("Invalid excitation type")
    end
end

# The band gap is given by the minimum of "particle excitation energy" + "hole excitation energy".