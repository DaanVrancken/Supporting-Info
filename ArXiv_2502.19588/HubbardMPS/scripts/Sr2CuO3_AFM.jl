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

# Provide Schmidt cut value as argument (s=5.0 is already considered good)
s = parse(Float64, ARGS[1])

P = 1;
Q = 1;
bond_dim = 20;

t = [0.4855, 0.0770, 0.0182]
mu = 0.1303
U = [3.3908, 1.0493]
J = [0.0312]

model = hf.OB_Sim(t, U, mu, J, P, Q, s, bond_dim; code = name*"$s", spin=false, U13=[-0.0332]);


########################
# COMPUTE GROUNDSTATES #
########################

dictionary = hf.compute_groundstate(model);
ψ₀ = dictionary["groundstate"];
H = dictionary["ham"];
E0 = expectation_value(ψ₀, H);
E = sum(real(E0))./length(H);
println("Groundstate energy: $E")
println("Bond dimension: $(hf.dim_state(ψ₀))")

Nup, Ndown = hf.density_spin(model)
println("Spin up: ", Nup)
println("Spin down: ", Ndown)
