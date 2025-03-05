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

# Provide Schmidt cut value as FIRST ARGUMENT (s=4.5 is already considered good)
s = parse(Float64, ARGS[1])

P = 1;                          # Filling P/Q
Q = 1;
bond_dim = 20;                  # Initial bond dimension

# extract parameters from params.jl file in folder
# change "tPA1" to "tPA2" or "tPA3" for other geometries
path = projectdir("data", "params", "tPA1", "params.jl")

t,U,J,U13,U13_IS,U112,U1111 = hf.extract_params(path; range_u=4, range_t=4, range_J=4, range_U13=4, r_112=4, r_1111=4)

t[1,8] = 0.0

function interaction_cutoff_tPA(U,J,U13_IS,U112,U1111,R=1)
    Bands = 2

    U_cut = copy(U)
    U_cut[1,(Bands+R):end] .= 0.0
    U_cut[2,(Bands+R+1):end] .= 0.0

    J_cut = copy(J)
    J_cut[1,(Bands+R):end] .= 0.0
    J_cut[2,(Bands+R+1):end] .= 0.0

    U13_IS_cut = copy(U13_IS)
    for i in 1:4
        U13_IS_cut[1,(R):end,i] .= 0.0
        U13_IS_cut[2,(R+1):end,i] .= 0.0
    end

    U112_cut = copy(U112)
    for key in keys(U112_cut)
        dist = maximum(key) - minimum(key)
        if dist > R
            delete!(U112_cut, key)
        end
    end

    U1111_cut = copy(U1111)
    for key in keys(U1111_cut)
        dist = maximum(key) - minimum(key)
        if dist > R
            delete!(U1111_cut, key)
        end
    end

    return U_cut, J_cut, U13_IS_cut, U112_cut, U1111_cut
end

# Input range (integer from 1 to 4) as SECOND ARGUMENT
R = Int(parse(Float64,ARGS[2]))
U,J,U13_IS,U112,U1111 = interaction_cutoff_tPA(U,J,U13_IS,U112,U1111,R)

# remove parameters of order of meV
for i in 1:2
    for j in 1:4
        if abs(J[i,j+2]) < 0.01
            J[i,j+2] = 0.0
        end
        for k in 1:4
            if abs(U13_IS[i,j,k]) < 0.01
                U13_IS[i,j,k] = 0.0
            end
        end
    end
end
if abs(J[2,7]) < 0.01
    J[2,7] = 0.0
end

for key in keys(U112)
    if abs(U112[key]) < 0.01 || 8 in key || (1 in key && 7 in key)
        delete!(U112, key)
    end
end

for key in keys(U1111)
    if abs(U1111[key]) < 0.01 || 8 in key || (1 in key && 7 in key)
        delete!(U1111, key)
    end
end

model = hf.MB_Sim(t, U, J, U13, P, Q, s, bond_dim; code=name*"U13_$(s)_$R",U13_IS=U13_IS, U112=U112, U1111=U1111);


#######################
# COMPUTE GROUNDSTATE #
#######################

Force = false
code = get(model.kwargs, :code, "")

if !isdir(datadir("jld2",code)) || Force
    if !isdir(datadir("jld2"))
        mkdir(datadir("jld2"))
    end
    println("Computing ground state...")
    dictionary = hf.compute_groundstate(model; tol=1e-5);
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

if length(ARGS) > 2
    resolution = 5;
    momenta = range(0, π, resolution);
    nums = 1;

    solver = Arnoldi(;krylovdim=25,tol=1e-5,eager=true)
    charges = [1,1/2,1]
    sector1 = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2]) ⊠ U1Irrep(charges[3]*Q)
    sector2 = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2]) ⊠ U1Irrep(-charges[3]*Q)
    # Provide as THIRD ARGUMENT "hole" or "particle" to compute the corresponding excitation
    if ARGS[3] == "hole"
        println("Computing hole excitation...")
        E1,_ = excitations(H, QuasiparticleAnsatz(solver), momenta, ψ₀; num=nums, sector=sector1)
        println("hole excitation: ", E1)
    elseif ARGS[3] == "particle"
        println("Computing particle excitation...")
        E2,_ = excitations(H, QuasiparticleAnsatz(solver), momenta, ψ₀; num=nums, sector=sector2)
        println("particle excitation: ", E2)
    else
        error("Invalid excitation type")
    end
end

# The band gap is given by the minimum of "particle excitation energy" + "hole excitation energy".