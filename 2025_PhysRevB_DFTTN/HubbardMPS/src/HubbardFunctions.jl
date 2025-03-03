module HubbardFunctions

# using BLISBLAS

export OB_Sim, MB_Sim, OBC_Sim, MBC_Sim
export produce_groundstate, produce_excitations, produce_bandgap, produce_TruncState
export dim_state, density_spin, density_state, plot_excitations, plot_spin

using DrWatson
using ThreadPinning
using Base.Threads
using LinearAlgebra
using MPSKit, MPSKitModels
using TensorKit
using KrylovKit
using DataFrames
using Plots
using Plots.PlotMeasures
using TensorOperations
using JLD2

function __init__()
    LinearAlgebra.BLAS.set_num_threads(1)
    if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "JOB_ID") || haskey(ENV, "PBS_JOBID")
        # Running on remote cluster
        ThreadPinning.pinthreads(:affinitymask)
    else
        # Running locally
        ThreadPinning.pinthreads(:cores)
    end
    MPSKit.Defaults.set_scheduler!(:greedy)   # serial -> disable multithreading, greedy -> greedy load-balancing, dynamic -> moderate load-balancing
    println("Running on $(Threads.nthreads()) threads.")
end

function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,SU2Irrep,U1Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×SU₂×U₁]$(parts)"
end

function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,U1Irrep,U1Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×U₁×U₁]$(parts)"
end

function Base.string(s::TensorKit.ProductSector{Tuple{FermionParity,SU2Irrep}})
    parts = map(x -> sprint(show, x; context=:typeinfo => typeof(x)), s.sectors)
    return "[fℤ₂×SU₂]$(parts)"
end

abstract type Simulation end
name(s::Simulation) = string(typeof(s))

"""
    OB_Sim(t::Vector{Float64}, u::Vector{Float64}, μ=0.0, J::Vector{Float64}, P=1, Q=1, svalue=2.0, bond_dim=50, period=0; kwargs...)

Construct a parameter set for a 1D one-band Hubbard model with a fixed number of particles.

# Arguments
- `t`: Vector in which element ``n`` is the value of the hopping parameter of distance ``n``. The first element is the nearest-neighbour hopping.
- `u`: Vector in which element ``n`` is the value of the Coulomb interaction with site at distance ``n-1``. The first element is the on-site interaction.
- `J`: Vector in which element ``n`` is the value of the exchange interaction with site at distance ``n``. The first element is the nearest-neighbour exchange.
- `µ`: The chemical potential.
- `P`,`Q`: The ratio `P`/`Q` defines the number of electrons per site, which should be larger than 0 and smaller than 2.
- `svalue`: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.
- `bond_dim`: The maximal bond dimension used to initialize the state.
- `Period`: Perform simulations on a helix with circumference `Period`. Value 0 corresponds to an infinite chain.

Put the optional argument `spin=true` to perform spin-dependent calculations.
"""
struct OB_Sim <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Float64
    J::Vector{Float64}
    P::Int64
    Q::Int64
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function OB_Sim(t::Vector{Float64}, u::Vector{Float64}, μ::Float64=0.0, P::Int64=1, Q::Int64=1, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, [0.0], P, Q, svalue, bond_dim, period, kwargs)
    end
    function OB_Sim(t::Vector{Float64}, u::Vector{Float64}, μ::Float64=0.0, J::Vector{Float64}=[0.0], P::Int64=1, Q::Int64=1, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, J, P, Q, svalue, bond_dim, period, kwargs)
    end
end
name(::OB_Sim) = "OB"

"""
    MB_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, P=1, Q=1, svalue=2.0, bond_dim=50; kwargs...)

Construct a parameter set for a 1D B-band Hubbard model with a fixed number of particles.

# Arguments
- `t`: Bx(nB) matrix in which element ``(i,j)`` is the hopping parameter from band ``i`` to band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... hopping matrices are concatenated horizontally.
- `u`: Bx(nB) matrix in which element ``(i,j)`` is the Coulomb repulsion ``U_{ij}=U_{iijj}`` between band ``i`` and band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally.
- `J`: Bx(nB) matrix in which element ``(i,j)`` is the exchange ``J_{ij}=U_{ijji}=U_{ijij}`` between band ``i`` and band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally. The diagonal terms of the on-site matrix are ignored.
- `U13`: BxB matrix in which element ``(i,j)`` is the parameter ``U_{ijjj}=U_{jijj}=U_{jjij}=U_{jjji}`` between band ``i`` and band ``j``. Only on-site. The diagonal terms of the on-site matrix are ignored. This argument is optional.
- `P`,`Q`: The ratio `P`/`Q` defines the number of electrons per site, which should be larger than 0 and smaller than 2.
- `svalue`: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.
- `bond_dim`: The maximal bond dimension used to initialize the state.

Put the optional argument 'spin=true' to perform spin-dependent calculations. 

U13 inter-site, Uijkk, and Uijkl can be inserted using kwargs.

Use the optional argument `name` to assign a name to the model. 
This is used to destinguish between different parameter sets: Wrong results could be loaded or overwritten if not used consistently!!!
"""
struct MB_Sim <: Simulation
    t::Matrix{Float64}                        #convention: number of bands = number of rows, BxB for on-site + Bx(B*range) matrix for IS
    u::Matrix{Float64}                        #convention: BxB matrix for OS (with OB on diagonal) + Bx(B*range) matrix for IS
    J::Matrix{Float64}                        #convention: BxB matrix for OS (with OB zeros) + Bx(B*range) matrix for IS
    U13::Matrix{Float64}                      #Matrix with iiij, iiji... parameters. Same convention.
    P::Int64
    Q::Int64
    svalue::Float64
    bond_dim::Int64
    kwargs
    function MB_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, P=1, Q=1, svalue=2.0, bond_dim = 50; kwargs...)
        Bands,_ = size(t)
        return new(t, u, J, zeros(Bands,Bands), P, Q, svalue, bond_dim, kwargs)
    end
    function MB_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, P=1, Q=1, svalue=2.0, bond_dim = 50; kwargs...)
        return new(t, u, J, U13, P, Q, svalue, bond_dim, kwargs)
    end
end
name(::MB_Sim) = "MB"

"""
    OBC_Sim(t::Vector{Float64}, u::Vector{Float64}, μf::Float64, svalue=2.0, bond_dim=50, period=0; mu=true, kwargs...)

Construct a parameter set for a 1D one-band Hubbard model with the number of particles determined by a chemical potential.

# Arguments
- `t`: Vector in which element ``n`` is the value of the hopping parameter of distance ``n``. The first element is the nearest-neighbour hopping.
- `u`: Vector in which element ``n`` is the value of the Coulomb interaction with site at distance ``n-1``. The first element is the on-site interaction.
- `µf`: The chemical potential, if `mu=true`. Otherwise, the filling of the system. The chemical potential corresponding to the given filling is determined automatically.
- `svalue`: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.
- `bond_dim`: The maximal bond dimension used to initialize the state.
- `Period`: Perform simulations on a helix with circumference `Period`. Value 0 corresponds to an infinite chain.

Spin-dependent calculations are not yet implemented.
"""
struct OBC_Sim <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Union{Float64, Nothing}    # Imposed chemical potential
    f::Union{Float64, Nothing}    # Fraction indicating the filling
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function OBC_Sim(t, u, μf::Float64, svalue=2.0, bond_dim = 50, period = 0; mu=true, kwargs...)
        spin::Bool = get(kwargs, :spin, false)
        if spin
            error("Spin not implemented.")
        end
        if mu
            return new(t, u, μf, nothing, svalue, bond_dim, period, kwargs)
        else
            if 0 < μf < 2
                return new(t, u, nothing, μf, svalue, bond_dim, period, kwargs)
            else
                return error("Filling should be between 0 and 2.")
            end
        end
    end
end
name(::OBC_Sim) = "OBC"

# used to compute groundstates in µ iterations
struct OBC_Sim2 <: Simulation
    t::Vector{Float64}
    u::Vector{Float64}
    μ::Union{Float64, Nothing}    # Imposed chemical potential
    svalue::Float64
    bond_dim::Int64
    period::Int64
    kwargs
    function OBC_Sim2(t, u, μ::Float64, svalue=2.0, bond_dim = 50, period = 0; kwargs...)
        return new(t, u, μ, svalue, bond_dim, period, kwargs)
    end
end
name(::OBC_Sim2) = "OBC2"

"""
    MBC_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, svalue=2.0, bond_dim=50; kwargs...)

Construct a parameter set for a 1D ``B``-band Hubbard model with the number of particles determined by a chemical potential.

# Arguments
- `t`: ``B\\times nB`` matrix in which element ``(i,j)`` is the hopping parameter from band ``i`` to band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... hopping matrices are concatenated horizontally. The diagonal terms of the on-site matrix determine the filling.
- `u`: ``B\\times nB`` matrix in which element ``(i,j)`` is the Coulomb repulsion ``U_{ij}=U_{iijj}`` between band ``i`` and band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally.
- `J`: ``B\\times nB`` matrix in which element ``(i,j)`` is the exchange ``J_{ij}=U_{ijji}=U_{ijij}`` between band ``i`` and band ``j``. The on-site, nearest neighbour, next-to-nearest neighbour... matrices are concatenated horizontally. The diagonal terms of the on-site matrix are ignored.
- `U13`: ``B\\times B`` matrix in which element ``(i,j)`` is the parameter ``U_{ijjj}=U_{jijj}=U_{jjij}=U_{jjji}`` between band ``i`` and band ``j``. Only on-site. The diagonal terms of the on-site matrix are ignored. This argument is optional.
- `svalue`: The Schmidt truncation value, used to truncate in the iDMRG2 algorithm for the computation of the groundstate.
- `bond_dim`: The maximal bond dimension used to initialize the state.

Spin-dependent calculations are not yet implemented.

U13 inter-site, Uijkk, and Uijkl can be inserted using kwargs.

Use the optional argument `name` to assign a name to the model. 
This is used to destinguish between different parameter sets: Wrong results could be loaded or overwritten if not used consistently!!!
"""
struct MBC_Sim <: Simulation
    t::Matrix{Float64}                        #convention: number of bands = number of rows, BxB for on-site + Bx(B*range) matrix for IS
    u::Matrix{Float64}                        #convention: BxB matrix for OS (with OB on diagonal) + Bx(B*range) matrix for IS
    J::Matrix{Float64}                        #convention: BxB matrix for OS (with OB zeros) + Bx(B*range) matrix for IS
    U13::Matrix{Float64}                      #Matrix with iiij, iiji... parameters. Same convention.
    svalue::Float64
    bond_dim::Int64
    kwargs
    function MBC_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, svalue=2.0, bond_dim = 50; kwargs...)
        spin::Bool = get(kwargs, :spin, false)
        if spin
            error("Spin not implemented.")
        end
        Bands,_ = size(t)
        return new(t, u, J, zeros(Bands,Bands), svalue, bond_dim, kwargs)
    end
    function MBC_Sim(t::Matrix{Float64}, u::Matrix{Float64}, J::Matrix{Float64}, U13::Matrix{Float64}, svalue=2.0, bond_dim = 50; kwargs...)
        spin::Bool = get(kwargs, :spin, false)
        if spin
            error("Spin not implemented.")
        end
        return new(t, u, J, U13, svalue, bond_dim, kwargs)
    end
end
name(::MBC_Sim) = "MBC"


###############
# Hamiltonian #
###############

function SymSpace(P,Q,spin)
    if spin
        I = fℤ₂ ⊠ U1Irrep ⊠ U1Irrep
        Ps = Vect[I]((0, 0, -P) => 1, (0, 0, 2*Q-P) => 1, (1, 1, Q-P) => 1, (1, -1, Q-P) => 1)
    else
        I = fℤ₂ ⊠ SU2Irrep ⊠ U1Irrep
        Ps = Vect[I]((0, 0, -P) => 1, (0, 0, 2*Q-P) => 1, (1, 1 // 2, Q-P) => 1)
    end

    return I, Ps
end

function Hopping(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        Vup = Vect[I]((1, 1, Q) => 1)
        Vdown = Vect[I]((1, -1, Q) => 1)
    
        c⁺u = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vup)
        blocks(c⁺u)[I((1, 1, Q-P))] .= 1
        blocks(c⁺u)[I((0, 0, 2*Q-P))] .= -1
        cu = TensorMap(zeros, ComplexF64, Vup ⊗ Ps ← Ps)
        blocks(cu)[I((1, 1, Q-P))] .= 1
        blocks(cu)[I((0, 0, 2*Q-P))] .= 1
        
        c⁺d = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vdown)
        blocks(c⁺d)[I((1, -1, Q-P))] .= 1
        blocks(c⁺d)[I((0, 0, 2*Q-P))] .= 1
        cd = TensorMap(zeros, ComplexF64, Vdown ⊗ Ps ← Ps)
        blocks(cd)[I((1, -1, Q-P))] .= 1
        blocks(cd)[I((0, 0, 2*Q-P))] .= -1
    
        @planar twosite_up[-1 -2; -3 -4] := c⁺u[-1; -3 1] * cu[1 -2; -4]
        @planar twosite_down[-1 -2; -3 -4] := c⁺d[-1; -3 1] * cd[1 -2; -4]
        twosite = twosite_up + twosite_down
    else
        Vs = Vect[I]((1, 1 / 2, Q) => 1)

        c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
        blocks(c⁺)[I((1, 1 // 2, Q-P))] .= 1
        blocks(c⁺)[I((0, 0, 2*Q-P))] .= sqrt(2)

        c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
        blocks(c)[I((1, 1 / 2, Q-P))] .= 1
        blocks(c)[I((0, 0, 2*Q-P))] .= sqrt(2)

        @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    end

    return twosite
end

function OSInteraction(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 0, 2*Q-P))] .= 1
    else
        onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(onesite)[I((0, 0, 2*Q-P))] .= 1
    end

    return onesite
end

function Number(P,Q,spin)
    I, Ps = SymSpace(P,Q,spin)

    if spin
        n = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(n)[I((0, 0, 2*Q-P))] .= 2
        blocks(n)[I((1, 1, Q-P))] .= 1
        blocks(n)[I((1, -1, Q-P))] .= 1
    else
        n = TensorMap(zeros, ComplexF64, Ps ← Ps)
        blocks(n)[I((0, 0, 2*Q-P))] .= 2
        blocks(n)[I((1, 1 // 2, Q-P))] .= 1
    end

    return n
end

function SymSpace()
    I = fℤ₂ ⊠ SU2Irrep
    Ps = Vect[I]((0, 0) => 2, (1, 1 // 2) => 1)

    return I, Ps
end

function Hopping()
    I, Ps = SymSpace()
    Vs = Vect[I]((1, 1 / 2) => 1)

    c⁺ = TensorMap(zeros, ComplexF64, Ps ← Ps ⊗ Vs)
    blocks(c⁺)[I((1, 1 // 2))] = [1.0+0.0im 0.0+0.0im]
    blocks(c⁺)[I((0, 0))] = [0.0+0.0im; sqrt(2)+0.0im;;]

    c = TensorMap(zeros, ComplexF64, Vs ⊗ Ps ← Ps)
    blocks(c)[I((1, 1 // 2))] = [1.0+0.0im; 0.0+0.0im;;]
    blocks(c)[I((0, 0))] = [0.0+0.0im sqrt(2)+0.0im]

    @planar twosite[-1 -2; -3 -4] := c⁺[-1; -3 1] * c[1 -2; -4]
    
    return twosite
end

function OSInteraction()
    I, Ps = SymSpace()

    onesite = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(onesite)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 1.0] 

    return onesite
end

function Number()
    I, Ps = SymSpace()

    n = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(n)[I((0, 0))] = [0.0+0.0im 0.0; 0.0 2.0] 
    blocks(n)[I((1, 1 // 2))] .= 1.0

    return n
end

# ONEBAND #

function hamiltonian(simul::Union{OB_Sim,OBC_Sim2})
    t = simul.t
    u = simul.u
    μ = simul.μ
    if hasproperty(simul, :J)
        J = simul.J
        D_exc = length(J)
    end
    L = simul.period
    spin::Bool = get(simul.kwargs, :spin, false)
    U13::Vector{Float64} = get(simul.kwargs, :U13, [0.0])

    D_hop = length(t)
    D_int = length(u)
    D_U13 = length(U13)
    
    if hasproperty(simul, :P)
        P = simul.P
        Q = simul.Q
        if iseven(P)
            T = Q
        else 
            T = 2*Q
        end
        cdc = Hopping(P,Q,spin)    
        n = Number(P,Q,spin)
        OSI = OSInteraction(P,Q,spin)
    else
        T = 1
        cdc = Hopping()
        n = Number()
        OSI = OSInteraction()
    end

    twosite = cdc + cdc'
    onesite = u[1]*OSI - μ*n

    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    @tensor J1[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[-2 3; 2 -3]
    @tensor J2[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[3 -2; -3 2]
    @tensor C1[-1 -2; -3 -4] := cdc[-1 2; -3 -4] * cdc[-2 3; 3 2]
    @tensor C2[-1 -2; -3 -4] := cdc[-1 2; -3 4] * cdc[-2 4; 2 -4]

    C1 = C1 + C1'
    C2 = C2 + C2'
    
    H = @mpoham sum(onesite{i} for i in vertices(InfiniteChain(T)))
    if L == 0
        for range_hop in 1:D_hop
            h = @mpoham sum(-t[range_hop]*twosite{i,i+range_hop} for i in vertices(InfiniteChain(T)))
            H += h
        end
        for range_int in 2:D_int  # first element is on-site interaction
            h = @mpoham sum(u[range_int]*nn{i,i+(range_int-1)} for i in vertices(InfiniteChain(T)))
            H += h
        end
        if hasproperty(simul, :J)
            for range_exc in 1:D_exc
                h1 = @mpoham sum(J[range_exc]*J1{i,i+range_exc} for i in vertices(InfiniteChain(T)))
                h2 = @mpoham sum(0.5*J[range_exc]*J2{i,i+range_exc} + 0.5*J[range_exc]*J2{i+range_exc,i} for i in vertices(InfiniteChain(T)))
                H += h1 + h2
            end
        end
        if U13 != [0.0]
            for range_U13 in 1:D_U13
                h3 = @mpoham sum(0.5*U13[range_U13]*C1{i,i+range_U13} + 0.5*U13[range_U13]*C2{i,i+range_U13} for i in vertices(InfiniteChain(T)))
                h4 = @mpoham sum(0.5*U13[range_U13]*C1{i+range_U13,i} + 0.5*U13[range_U13]*C2{i+range_U13,i} for i in vertices(InfiniteChain(T)))
                H += h3 + h4
            end
        end
    elseif D_hop==1 && D_int==1
        h = @mpoham sum(-t[1]*twosite{i,i+1} -t[1]*twosite{i,i+L} for i in vertices(InfiniteChain(T)))
        H += h
    else
        return error("Extended models in 2D not implemented.")
    end

    return H
end

# MULTIBAND #

# t[i,j] gives the hopping of band i on one site to band j on the same site (i≠j)
function OS_Hopping(t,T,cdc)
    Bands,Bands2 = size(t)
    
    if Bands ≠ Bands2 || typeof(t) ≠ Matrix{Float64}
        @warn "t_OS is not a float square matrix."
    end
    for i in 1:Bands
        for j in (i+1):Bands
            if !(t[i,j] ≈ t'[i,j])
                @warn "t_OS is not Hermitian."
            end
        end
    end
    
    Lattice = InfiniteStrip(Bands,T*Bands)
        
    # Define necessary different indices of sites/orbitals in the lattice
    # Diagonal terms are taken care of in chem_pot
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(T*Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(-t[bi,bf]*cdc{Lattice[bf,site],Lattice[bi,site]} for (site, bi, bf) in Indices)
end

# t[i,j] gives the hopping of band i on one site to band j on the range^th next site
# parameter must be equal in both directions (1i->2j=2j->1i) to guarantee hermiticity
function IS_Hopping(t,range,T,cdc)
    Bands,Bands2 = size(t)
    if Bands ≠ Bands2 || typeof(t) ≠ Matrix{Float64}
        @warn "t_IS is not a float square matrix"
    end
    
    twosite = cdc + cdc'
    Lattice = InfiniteStrip(Bands,T*Bands)
        
    # Define necessary different indices of sites/orbitals in the lattice
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands^2)]
    
    return @mpoham sum(-t[bi,bf]*twosite{Lattice[bf,site+range],Lattice[bi,site]} for (site, bi, bf) in Indices)
end

# μ[i] gives the hopping of band i on one site to band i on the same site.
function Chem_pot(μ,T,n)
    Bands = length(μ)
    
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands)]
    
    return @mpoham sum(-μ[i]*n{Lattice[i,j]} for (j,i) in Indices)
end

# u[i] gives the interaction on band i
function OB_interaction(u,T,OSI)
    Bands = length(u)
    
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands)]
    
    return @mpoham sum(u[i]*OSI{Lattice[i,j]} for (j,i) in Indices)
end

# U[i,j] gives the direct interaction between band i on one site to band j on the same site. Averaged over U[i,j] and U[j,i]
function Direct_OS(U,T,n)
    Bands,Bands2 = size(U)
    
    if Bands ≠ Bands2 || typeof(U) ≠ Matrix{Float64}
        @warn "U_OS is not a float square matrix"
    end
    
    U_av = zeros(Bands,Bands2)
    for i in 2:Bands    
        for j in 1:(i-1)
            U_av[i,j] = 0.5*(U[i,j]+U[j,i])
        end
    end
    
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(T*Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(U_av[bi,bf]*nn{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices if U_av[bi,bf]≠0.0)
end

# J[i,j] gives the exchange interaction between band i on one site to band j on the same site.
function Exchange1_OS(J,T,cdc)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J_OS is not a float square matrix"
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = J[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band interaction is not taken into account in Exchange_OS."
    end
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[-2 3; 2 -3]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(T*Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)
end;

function Exchange2_OS(J,T,cdc)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J_OS is not a float square matrix"
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = J[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band interaction is not taken into account in Exchange_OS."
    end
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[3 -2; -3 2]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(T*Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)
end;

function Exchange_OS(J,T,cdc)
    return Exchange1_OS(J,T,cdc) + Exchange2_OS(J,T,cdc)
end;

function Uijjj_OS(U,T,cdc)
    Bands,Bands2 = size(U)
    
    if Bands ≠ Bands2 || typeof(U) ≠ Matrix{Float64}
        @warn "U13_OS is not a float square matrix"
    end
    diagonal = zeros(Bands,1)
    diagonal_zeros = zeros(Bands,1)
    for i in 1:Bands
        diagonal[i] = U[i,i]
    end
    if diagonal≠diagonal_zeros
        @warn "On-band interaction is not taken into account in Exchange_OS."
    end
    
    @tensor C1[-1 -2; -3 -4] := cdc[-1 2; -3 -4] * cdc[-2 3; 3 2]
    #@tensor C2[-1 -2; -3 -4] := cdc[-2 -1; 3 -3] * cdc[3 2; 2 -4]
    @tensor C2[-1 -2; -3 -4] := cdc[-1 2; -3 4] * cdc[-2 4; 2 -4]
    #@tensor C4[-1 -2; -3 -4] := cdc[1 -1; 3 -3] * cdc[-2 3; 1 -4]

    C1 = C1 + C1'
    C2 = C2 + C2'

    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) 
               for l in 1:(T*Bands^2) if div((l-1)%(Bands^2),Bands)+1 ≠ mod(l-1,Bands)+1]
    
    H = @mpoham sum(0.5*U[bi,bf]*C1{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)
    H += @mpoham sum(0.5*U[bi,bf]*C2{Lattice[bi,site],Lattice[bf,site]} for (site,bi,bf) in Indices)

    return H
end;

# V[i,j] gives the direct interaction between band i on one site to band j on the range^th next site.
function Direct_IS(V,range,T,n)
    Bands,Bands2 = size(V)
    
    if Bands ≠ Bands2 || typeof(V) ≠ Matrix{Float64}
        @warn "V is not a float square matrix"
    end
    
    @planar nn[-1 -2; -3 -4] := n[-1; -3] * n[-2; -4]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands^2)]
    
    return @mpoham sum(V[bi,bf]*nn{Lattice[bi,site],Lattice[bf,site+range]} for (site,bi,bf) in Indices)
end

# J[i,j] gives the exchange interaction between band i on one site to band j on the range^th next site.
function Exchange1_IS(J,range,T,cdc)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J_IS is not a float square matrix"
    end
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[-2 3; 2 -3]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands^2)]
    
    return @mpoham sum(J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site+range]} for (site,bi,bf) in Indices)    # operator has no direction
end;

function Exchange2_IS(J,range,T,cdc)
    Bands,Bands2 = size(J)
    
    if Bands ≠ Bands2 || typeof(J) ≠ Matrix{Float64}
        @warn "J_IS is not a float square matrix"
    end
    
    @tensor C4[-1 -2; -3 -4] := cdc[-1 2; 3 -4] * cdc[3 -2; -3 2]
    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands^2)]
    
    return @mpoham sum(0.5*J[bi,bf]*C4{Lattice[bi,site],Lattice[bf,site+range]} + 0.5*J[bi,bf]*C4{Lattice[bf,site+range],Lattice[bi,site]} for (site,bi,bf) in Indices) #operator has direction
end;

function Exchange_IS(J,range,T,cdc)
    return Exchange1_IS(J,range,T,cdc) + Exchange2_IS(J,range,T,cdc)
end;

# Four different matrices required: two for U13 and two for U31
function Uijjj_IS(U,range,T,cdc)
    Bands,Bands2,num = size(U)
    
    if Bands ≠ Bands2
        @warn "U13_IS is not a float square matrix"
    elseif num != 4
        # i = orbital 1 on site 0, j = orbital 2 on site "range"
        # index 1: Uijjj=Ujjji, index 2: Ujiii=Uiiij, index 3: Ujijj=Ujjij, index 4: Uijii=Uiiji
        error("U13_IS shoud be a BxBx4 array.")
    end
    
    @tensor C1[-1 -2; -3 -4] := cdc[-1 2; -3 -4] * cdc[-2 3; 3 2]
    #@tensor C2[-1 -2; -3 -4] := cdc[-2 -1; 3 -3] * cdc[3 2; 2 -4]
    @tensor C2[-1 -2; -3 -4] := cdc[-1 2; -3 4] * cdc[-2 4; 2 -4]
    #@tensor C4[-1 -2; -3 -4] := cdc[1 -1; 3 -3] * cdc[-2 3; 1 -4]

    C1 = C1 + C1'
    C2 = C2 + C2'

    Lattice = InfiniteStrip(Bands,T*Bands)
    
    Indices = [(div(l-1,Bands^2)+1, div((l-1)%(Bands^2),Bands)+1, mod(l-1,Bands)+1) for l in 1:(T*Bands^2)]
    
    H = @mpoham sum(0.5*U[bi,bf,1]*C1{Lattice[bi,site],Lattice[bf,site+range]} + 0.5*U[bi,bf,3]*C1{Lattice[bf,site+range],Lattice[bi,site]} for (site,bi,bf) in Indices) #operator has direction
    H += @mpoham sum(0.5*U[bi,bf,2]*C2{Lattice[bi,site],Lattice[bf,site+range]} + 0.5*U[bi,bf,4]*C2{Lattice[bf,site+range],Lattice[bi,site]} for (site,bi,bf) in Indices)

    return H
end;

function Uijkk(U::Dict{NTuple{4, Int64}, Float64},B,T,cdc)
    # input is dict with permutations i,j,k,l (in order Cdi Cdj Ck Cl, NOT Uijkl). Indices range over i,j,k,l = 1,...,r*B
    # At least one index in every tuple (i,j,k,l) has to be at site 0

    Ind1 = []
    Ind2 = []
    Ind3 = []
    for (i,j,k,l) in keys(U)
        if minimum((i,j,k,l)) > B
            error("At least one index in every tuple (i,j,k,l) has to be at site 0.")
        elseif length(unique((i, j, k, l))) != 3
            error("Two indices should be the same. Not more, not less.")
        end
        for site in 1:T
            if k==l
                push!(Ind1,(site,i,j,k,l))
            elseif j==k
                push!(Ind2,(site,i,j,k,l))
            elseif j==l
                push!(Ind3,(site,i,j,k,l))
            end
        end
    end

    @tensor C1[-1 -2 -3; -4 -5 -6] := cdc[-1 2; -4 -6] * cdc[-2 -3; -5 2]
    @tensor C2[-1 -2 -3; -4 -5 -6] := cdc[-1 -3; -4 -6] * cdc[-2 2; 2 -5]
    @tensor C3[-1 -2 -3; -4 -5 -6] := cdc[-1 2; -4 -5] * cdc[-2 -3; 2 -6]
    C1 = C1 + C1'
    C2 = C2 + C2'
    C3 = C3 + C3'

    Lattice = InfiniteStrip(B,T*B)
    
    @tensor init_operator[-1; -2] := cdc[-1 2; 2 -2]
    H = @mpoham sum(0.0*init_operator{i} for i in vertices(Lattice))
    if !isempty(Ind1)
        H += @mpoham sum(0.5*U[(i,j,k,l)]*C1{Lattice[mod(i-1,B)+1,site+(i-1)÷B],Lattice[mod(j-1,B)+1,site+(j-1)÷B],Lattice[mod(k-1,B)+1,site+(k-1)÷B]} for (site,i,j,k,l) in Ind1)
    end
    if !isempty(Ind2)
        H += @mpoham sum(U[(i,j,k,l)]*C2{Lattice[mod(i-1,B)+1,site+(i-1)÷B],Lattice[mod(j-1,B)+1,site+(j-1)÷B],Lattice[mod(l-1,B)+1,site+(l-1)÷B]} for (site,i,j,k,l) in Ind2)
    end
    if !isempty(Ind3)
        H += @mpoham sum(0.5*U[(i,j,k,l)]*C3{Lattice[mod(i-1,B)+1,site+(i-1)÷B],Lattice[mod(j-1,B)+1,site+(j-1)÷B],Lattice[mod(k-1,B)+1,site+(k-1)÷B]} for (site,i,j,k,l) in Ind3)
    end

    return H
end;

function Uijkl(U::Dict{NTuple{4, Int64}, Float64},B,T,cdc)
    # input is dict with permutations i,j,k,l (in order Cdi Cdj Ck Cl, NOT Uijkl). Indices range over i,j,k,l = 1,...,r*B
    # At least one index in every tuple (i,j,k,l) has to be at site 0

    Ind = []
    for (i,j,k,l) in keys(U)
        if minimum((i,j,k,l)) > B
            error("At least one index in every tuple (i,j,k,l) has to be at site 0.")
        elseif length(unique((i, j, k, l))) != 4
            error("All indices must be different.")
        elseif !(U[(i,j,k,l)] ≈ U[(l,k,j,i)])
            @warn("U1111 is not Hermitian.")
        end
        for site in 1:T
            push!(Ind,(site,i,j,k,l))
        end
    end

    @tensor C[-1 -2 -3 -4; -5 -6 -7 -8] := cdc[-1 -2; -5 -6] * cdc[-3 -4; -7 -8]

    Lattice = InfiniteStrip(B,T*B)

    @tensor init_operator[-1; -2] := cdc[-1 2; 2 -2]
    H = @mpoham sum(0.0*init_operator{i} for i in vertices(Lattice))
    if !isempty(Ind)
        H += @mpoham sum(0.5*U[(i,j,k,l)]*C{Lattice[mod(i-1,B)+1,site+(i-1)÷B],Lattice[mod(l-1,B)+1,site+(l-1)÷B],Lattice[mod(j-1,B)+1,site+(j-1)÷B,],Lattice[mod(k-1,B)+1,site+(k-1)÷B]} for (site,i,j,k,l) in Ind)
    end

    return H
end

function hamiltonian(simul::Union{MB_Sim, MBC_Sim})
    t = simul.t
    u = simul.u
    J = simul.J
    U13_OS = simul.U13
    U112::Dict{NTuple{4, Int64}, Float64} = get(simul.kwargs, :U112, Dict{Tuple{Int, Int, Int, Int}, Float64}())
    U1111::Dict{NTuple{4, Int64}, Float64} = get(simul.kwargs, :U1111, Dict{Tuple{Int, Int, Int, Int}, Float64}())
    spin::Bool = get(simul.kwargs, :spin, false)

    Bands,width_t = size(t)
    Bands1,width_u = size(u)
    Bands2,width_J = size(J)
    Bands3,_ = size(U13_OS)
    U13_IS::Array{Float64, 3} = get(simul.kwargs, :U13_IS, zeros(Bands,Bands,4))
    if !(Bands == Bands1 == Bands2 == Bands3 == size(U13_IS)[1])
        return error("Number of bands is incosistent.")
    end

    if hasproperty(simul, :P)
        P = simul.P
        Q = simul.Q
        if iseven(P)
            T = Q
        else 
            T = 2*Q
        end
        cdc = Hopping(P,Q,spin)
        OSI = OSInteraction(P,Q,spin)
        n = Number(P,Q,spin)
    else
        T = 1
        cdc = Hopping()
        OSI = OSInteraction()
        n = Number()
    end

    Range_t = Int((width_t-Bands)/Bands)
    Range_u = Int((width_u-Bands)/Bands)
    Range_J = Int((width_J-Bands)/Bands)
    Range_U13 = Int((size(U13_IS)[2])/Bands)

    # Define matrices
    u_OB = zeros(Bands)
    for i in 1:Bands
        u_OB[i] = u[i,i]
    end
    if u_OB == zeros(Bands)
        @warn "No on-band interaction found. This may lead to too low contributions of other Hamiltonian terms."
    end
    t_OS = t[:,1:Bands]
    μ = zeros(Bands)
    for i in 1:Bands
        μ[i] = t_OS[i,i]
    end
    u_OS = u[:,1:Bands]
    for i in 1:Bands
        u_OS[i,i] = 0.0
    end
    J_OS = J[:,1:Bands]

    # Implement Hamiltonian OB
    H_total = OB_interaction(u_OB,T,OSI)

    if μ != zeros(Bands)
        H_total += Chem_pot(μ,T,n)
    end

    # Implement Hamiltonian OS
    for (m,o,f) in [(t_OS,cdc,OS_Hopping),(u_OS,n,Direct_OS),(J_OS,cdc,Exchange_OS),(U13_OS,cdc,Uijjj_OS)]
        if m != zeros(Bands,Bands)
            H_total += f(m,T,o)
        end
    end

    # Implement Hamiltonian IS
    for (m,range,o,f) in [(t,Range_t,cdc,IS_Hopping),(u,Range_u,n,Direct_IS),(J,Range_J,cdc,Exchange_IS),(U13_IS,Range_U13,cdc,Uijjj_IS)]
        for i in 1:range
            if m != U13_IS
                M = m[:,(Bands*i+1):(Bands*(i+1))]
                ZERO = zeros(Bands,Bands)
            else
                M = m[:,(Bands*(i-1)+1):(Bands*i),:]
                ZERO = zeros(Bands,Bands,4)
            end
            if M != ZERO
                H_total += f(M,i,T,o)
            end
        end
    end

    if !isempty(U112)
        H_total += Uijkk(U112::Dict{NTuple{4, Int64}, Float64},Bands,T,cdc)
    end

    if !isempty(U1111)
        H_total += Uijkl(U1111::Dict{NTuple{4, Int64}, Float64},Bands,T,cdc)
    end

    return H_total
end


###############
# Groundstate #
###############

function initialize_mps(operator, P::Int64, max_dimension::Int64, spin::Bool)
    Ps = physicalspace.(parent(operator))
    L = length(Ps)
    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    if !spin
        Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((0,0,0)=>1)     # find maximal virtual space
        for i in 0:1
            for j in 0:1//2:3
                for k in -(L*P):1:(L*P)
                    Vmax = Vect[(FermionParity ⊠ Irrep[SU₂] ⊠ Irrep[U₁])]((i,j,k)=>max_dimension) ⊕ Vmax
                end
            end
        end
    else
        Vmax = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((0,0,0)=>1)
        for i in 0:1
            for j in -L:1:L
                for k in -(L*P):1:(L*P)
                    Vmax = Vect[(FermionParity ⊠ Irrep[U₁] ⊠ Irrep[U₁])]((i,j,k)=>max_dimension) ⊕ Vmax
                end
            end
        end
    end

    V_max = copy(V)

    for i in 1:length(V_right)
        V_max[i] = Vmax
    end

    V_trunc = TensorKit.infimum.(V,V_max)

    return InfiniteMPS(Ps, V_trunc)
end

function initialize_mps(operator, max_dimension::Int64)
    Ps = physicalspace.(parent(operator))

    V_right = accumulate(fuse, Ps)
    
    V_l = accumulate(fuse, dual.(Ps); init=one(first(Ps)))
    V_left = reverse(V_l)
    len = length(V_left)
    step = length(V_left)-1
    V_left = [view(V_left,len-step+1:len); view(V_left,1:len-step)]   # same as circshift(V_left,1)

    V = TensorKit.infimum.(V_left, V_right)

    Vmax = Vect[(FermionParity ⊠ Irrep[SU₂])]((0,0)=>1)     # find maximal virtual space

    for i in 0:1
        for j in 0:1//2:3
            Vmax = Vect[(FermionParity ⊠ Irrep[SU₂])]((i,j)=>max_dimension) ⊕ Vmax
        end
    end

    V_max = copy(V)      # if no copy(), V will change along when V_max is changed

    for i in 1:length(V_right)
        V_max[i] = Vmax
    end

    V_trunc = TensorKit.infimum.(V,V_max)

    return InfiniteMPS(Ps, V_trunc)
end

function compute_groundstate(simul::Union{OB_Sim, MB_Sim, OBC_Sim2, MBC_Sim}; tol::Float64=1e-6, verbosity::Int64=0, maxiter::Int64=1000, init_state=nothing)
    H = hamiltonian(simul)
    spin::Bool = get(simul.kwargs, :spin, false)

    if isnothing(init_state)
        if hasproperty(simul, :P)
            ψ₀ = initialize_mps(H,simul.P,simul.bond_dim,spin)
        else
            ψ₀ = initialize_mps(H,simul.bond_dim)
        end
    else
        ψ₀ = init_state
    end
    
    schmidtcut = 10.0^(-simul.svalue)
    
    if length(H) > 1
        ψ₀, envs, = find_groundstate(ψ₀, H, IDMRG2(; trscheme=truncbelow(schmidtcut), tol=tol, verbosity=verbosity))
    else
        ψ₀, envs, = find_groundstate(ψ₀, H, VUMPS(; tol=max(tol, schmidtcut/10), verbosity=verbosity))
        ψ₀ = changebonds(ψ₀, SvdCut(; trscheme=truncbelow(schmidtcut)))
        χ = sum(i -> dim(left_virtualspace(ψ₀, i)), 1:length(H))
        for i in 1:maxiter
            ψ₀, envs = changebonds(ψ₀, H, VUMPSSvdCut(;trscheme=truncbelow(schmidtcut)))
            ψ₀, = find_groundstate(ψ₀, H, VUMPS(; tol=max(tol, schmidtcut / 10), verbosity=verbosity), envs)
            ψ₀ = changebonds(ψ₀, SvdCut(; trscheme=truncbelow(schmidtcut)))
            χ′ = sum(i -> dim(left_virtualspace(ψ₀, i)), 1:length(H))
            isapprox(χ, χ′; rtol=0.05) && break
            χ = χ′
        end
    end
    
    alg = VUMPS(; maxiter=maxiter, tol=tol, verbosity=verbosity) &
        GradientGrassmann(; maxiter=maxiter, tol=tol, verbosity=verbosity)
    ψ, envs, δ = find_groundstate(ψ₀, H, alg)
    
    return Dict("groundstate" => ψ, "environments" => envs, "ham" => H, "delta" => δ, "config" => simul)
end

function compute_groundstate(simul::OBC_Sim; tol::Float64=1e-6, verbosity::Int64=0, maxiter::Int64=1000)
    verbosity_mu = get(simul.kwargs, :verbosity_mu, 0)
    t = simul.t
    u = simul.u
    s = simul.svalue
    bond_dim=simul.bond_dim 
    period = simul.period
    kwargs = simul.kwargs

    if simul.μ !== nothing
        simul2 = OBC_Sim2(t,u,simul.μ,s,bond_dim,period;kwargs)
        dictionary = compute_groundstate(simul2; tol=tol, verbosity=verbosity, maxiter=maxiter);
        dictionary["μ"] = simul.μ
    else 
        f = simul.f
        tol_mu = get(kwargs, :tol_mu, 1e-8)
        maxiter_mu = get(kwargs, :maxiter_mu, 20)
        step_size = get(kwargs, :step_size, 1.0)
        flag = false

        lower_bound = get(simul.kwargs, :lower_mu, 0.0)
        upper_bound = get(simul.kwargs, :upper_mu, 0.0)
        mid_point = (lower_bound + upper_bound)/2
        i = 1

        simul2 = OBC_Sim2(t,u,lower_bound,s,bond_dim,period;kwargs)
        dictionary_l = compute_groundstate(simul2; tol=tol, verbosity=verbosity, maxiter=maxiter);
        dictionary_u = deepcopy(dictionary_l)
        dictionary_sp = deepcopy(dictionary_l)
        while i<=maxiter_mu
            if abs(density_state(dictionary_u["groundstate"]) - f) < tol_mu
                flag=true
                dictionary_sp = deepcopy(dictionary_u)
                mid_point = upper_bound
                break
            elseif abs(density_state(dictionary_l["groundstate"]) - f) < tol_mu
                flag=true
                dictionary_sp = deepcopy(dictionary_l)
                mid_point = lower_bound
                break
            elseif density_state(dictionary_u["groundstate"]) < f
                lower_bound = copy(upper_bound)
                upper_bound += step_size
                simul2 = OBC_Sim2(t,u,upper_bound,s,bond_dim,period;kwargs)
                dictionary_u = compute_groundstate(simul2; tol=tol, verbosity=verbosity, maxiter=maxiter)
            elseif density_state(dictionary_l["groundstate"]) > f
                upper_bound = copy(lower_bound)
                lower_bound -= step_size
                simul2 = OBC_Sim2(t,u,lower_bound,s,bond_dim,period;kwargs)
                dictionary_l = compute_groundstate(simul2; tol=tol, verbosity=verbosity, maxiter=maxiter)
            else
                break
            end
            verbosity_mu>0 && @info "Iteration μ: $i => Lower bound: $lower_bound; Upper bound: $upper_bound"
            i+=1
        end
        if upper_bound>0.0
            value = "larger"
            dictionary = dictionary_u
        else
            value = "smaller"
            dictionary = dictionary_l
        end
        if i>maxiter_mu
            max_value = (i-1)*step_size
            @warn "The chemical potential is $value than: $max_value. Increase the stepsize."
        end

        while abs(density_state(dictionary["groundstate"]) - f)>tol_mu && i<=maxiter_mu && !flag
            mid_point = (lower_bound + upper_bound)/2
            simul2 = OBC_Sim2(t,u,mid_point,s,bond_dim,period;kwargs)
            dictionary = compute_groundstate(simul2)
            if density_state(dictionary["groundstate"]) < f
                lower_bound = copy(mid_point)
            else
                upper_bound = copy(mid_point)
            end
            verbosity_mu>0 && @info "Iteration μ: $i => Lower bound: $lower_bound; Upper bound: $upper_bound"
            i+=1
        end
        if i>maxiter_mu
            @warn "The chemical potential lies between $lower_bound and $upper_bound, but did not converge within the tolerance. Increase maxiter_mu."
        else
            verbosity_mu>0 && @info "Final chemical potential = $mid_point"
        end

        if flag
            dictionary = dictionary_sp
        end

        dictionary["μ"] = mid_point
    end

    return dictionary
end

"""
    produce_groundstate(model::Simulation; force::Bool=false)

Compute or load groundstate of the `model`. If `force=true`, overwrite existing calculation.
"""
function produce_groundstate(simul::Union{MB_Sim, MBC_Sim}; force::Bool=false)
    code = get(simul.kwargs, :code, "")
    S = "nospin_"
    spin::Bool = get(simul.kwargs, :spin, false)
    if spin
        S = "spin_"
    end

    data, _ = produce_or_load(compute_groundstate, simul, datadir("sims", name(simul)); prefix="groundstate_"*S*code, force=force)
    return data
end

function produce_groundstate(simul::Union{OB_Sim, OBC_Sim}; force::Bool=false)
    t = simul.t 
    u = simul.u
    if hasproperty(simul, :J)
        J = simul.J
    else
        J = 0
    end
    S_spin = "nospin_"
    spin::Bool = get(simul.kwargs, :spin, false)
    if spin
        S_spin = "spin_"
    end
    S = "groundstate_"*S_spin*"t$(t)_u$(u)_J$(J)"
    S = replace(S, ", " => "_")
    data, _ = produce_or_load(compute_groundstate, simul, datadir("sims", name(simul)); prefix=S, force=force)
    return data
end


###############
# Excitations #
###############

function compute_excitations(simul::Simulation, momenta, nums::Int64; 
                                    charges::Vector{Float64}=[0,0.0,0], 
                                    trunc_dim::Int64=0, trunc_scheme::Int64=0, DW = false, shift=1,
                                    solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true))
    if trunc_dim<0
        return error("Trunc_dim should be a positive integer.")
    end
    spin::Bool = get(simul.kwargs, :spin, false)

    if hasproperty(simul, :Q)
        Q = simul.Q
        if !spin
            sector = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2]) ⊠ U1Irrep(charges[3]*Q)
        else
            sector = fℤ₂(charges[1]) ⊠ U1Irrep(charges[2]) ⊠ U1Irrep(charges[3]*Q)
        end
    else
        sector = fℤ₂(charges[1]) ⊠ SU2Irrep(charges[2])
    end

    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    if trunc_dim==0
        envs = dictionary["environments"]
    else
        dict_trunc = produce_TruncState(simul, trunc_dim; trunc_scheme=trunc_scheme)
        ψ = dict_trunc["ψ_trunc"]
        envs = dict_trunc["envs_trunc"]
    end
    if DW
        ψ_s = circshift(ψ, shift)
        envs_s = environments(ψ_s, H);
        Es, qps = excitations(H, QuasiparticleAnsatz(solver), momenta./length(H), ψ, envs, ψ_s, envs_s; num=nums, sector=sector)
    else
        Es, qps = excitations(H, QuasiparticleAnsatz(solver), momenta./length(H), ψ, envs; num=nums, sector=sector)
    end
    
    return Dict("Es" => Es, "qps" => qps, "momenta" => momenta)
end

"""
    produce_excitations(model::Simulation, momenta, nums::Int64; force::Bool=false, charges::Vector{Float64}=[0,0.0,0], kwargs...)

Compute or load quasiparticle excitations of the desired `model`.

# Arguments
- `model`: Model for which excitations are sought.
- `momenta`: Momenta of the quasiparticle excitations.
- `nums`: Number of excitations.
- `force`: If true, overwrite existing calculation.
- `charges`: Charges of the symmetry sector of the excitations.
"""
function produce_excitations(simul::Simulation, momenta, nums::Int64; 
                                    force::Bool=false, charges::Vector{Float64}=[0,0.0,0], 
                                    trunc_dim::Int64=0, trunc_scheme::Int64=0, 
                                    solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true))
    spin::Bool = get(simul.kwargs, :spin, false)
    S = ""
    if typeof(momenta)==Float64
        momenta_string = "_mom=$momenta"
    else
        momenta_string = "_mom=$(first(momenta))to$(last(momenta))div$(length(momenta))"
    end
    if hasproperty(simul, :Q)
        if !spin
            charge_string = "f$(Int(charges[1]))su$(charges[2])u$(Int(charges[3]))"
        else
            charge_string = "f$(Int(charges[1]))u$(charges[2])u$(Int(charges[3]))"
            S = "spin_"
        end
    else
        charge_string = "f$(Int(charges[1]))su$(charges[2])"
    end

    code = get(simul.kwargs, :code, "")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix="excitations_"*S*code*"_nums=$nums"*"charges="*charge_string*momenta_string*"_trunc=$trunc_dim", force=force) do cfg
        return compute_excitations(cfg, momenta, nums; charges=charges, trunc_dim=trunc_dim, trunc_scheme=trunc_scheme, solver=solver)
    end
    return data
end

"""
    produce_bandgap(model::Union{OB_Sim, MB_Sim}; resolution::Int64=5, force::Bool=false)

Compute or load the band gap of the desired model.
"""
function produce_bandgap(simul::Union{OB_Sim, MB_Sim}; resolution::Int64=5, force::Bool=false)
    momenta = range(0, π, resolution)
    Q = simul.Q
    spin::Bool = get(simul.kwargs, :spin, false)

    if spin
        error("Band gap for spin systems not implemented.")
    end

    Exc_hole = produce_excitations(simul, momenta, 1; force=force, charges=[1,1/2,-Q])
    Exc_elec = produce_excitations(simul, momenta, 1; force=force, charges=[1,1/2,Q])

    E_hole = real(Exc_hole["Es"])
    E_elec = real(Exc_elec["Es"])

    E_tot = E_hole + E_elec

    gap, k = findmin(E_tot[:,1])

    if k != 1
        @warn "Indirect band gap! Higher resolution might be required."
    end

    return gap, momenta[k]
end

function produce_domainwalls(simul::Simulation, momenta, nums::Int64; 
                                    force::Bool=false, charges::Vector{Float64}=[0,0.0,1], 
                                    trunc_dim::Int64=0, trunc_scheme::Int64=0, shift=1,
                                    solver=Arnoldi(;krylovdim=30,tol=1e-6,eager=true))
    spin::Bool = get(simul.kwargs, :spin, false)
    S = ""
    if typeof(momenta)==Float64
        momenta_string = "_mom=$momenta"
    else
        momenta_string = "_mom=$(first(momenta))to$(last(momenta))div$(length(momenta))"
    end
    if hasproperty(simul, :Q)
        if !spin
            charge_string = "f$(Int(charges[1]))su$(charges[2])u$(Int(charges[3]))"
        else
            charge_string = "f$(Int(charges[1]))u$(charges[2])u$(Int(charges[3]))"
            S = "spin_"
        end
    else
        charge_string = "f$(Int(charges[1]))su$(charges[2])"
    end

    code = get(simul.kwargs, :code, "")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix="DWs_"*S*code*"_nums=$nums"*"charges="*charge_string*momenta_string*"_trunc=$trunc_dim", force=force) do cfg
        return compute_excitations(cfg, momenta, nums; charges=charges, trunc_dim=trunc_dim, trunc_scheme=trunc_scheme, DW=true, shift=shift, solver=solver)
    end
    return data
end


##############
# Truncation #
##############

function TruncState(simul::Simulation, trunc_dim::Int64; trunc_scheme::Int64=0)
    if trunc_dim<=0
        return error("trunc_dim should be a positive integer.")
    end
    if trunc_scheme!=0 && trunc_scheme!=1
        return error("trunc_scheme should be either 0 (VUMPSSvdCut) or 1 (SvdCut).")
    end

    dictionary = produce_groundstate(simul)
    ψ = dictionary["groundstate"]
    H = dictionary["ham"]
    if trunc_scheme==0
        ψ, envs = changebonds(ψ,H,VUMPSSvdCut(; trscheme=truncdim(trunc_dim)))
    else
        ψ, envs = changebonds(ψ,H,SvdCut(; trscheme=truncdim(trunc_dim)))
    end
    return  Dict("ψ_trunc" => ψ, "envs_trunc" => envs)
end

"""
    produce_truncstate(model::Simulation, trunc_dim::Int64; trunc_scheme::Int64=0, force::Bool=false)

Compute or load a truncated approximation of the groundstate.

# Arguments
- `model`: Model for which the groundstate is to be truncated.
- `trunc_dim`: Maximal bond dimension of the truncated state.
- `trunc_scheme`: Scheme to perform the truncation. 0 = VUMPSSvdCut. 1 = SvdCut.
- `force`: If true, overwrite existing calculation.
"""
function produce_TruncState(simul::Simulation, trunc_dim::Int64; trunc_scheme::Int64=0, force::Bool=false)
    code = get(simul.kwargs, :code, "")
    data, _ = produce_or_load(simul, datadir("sims", name(simul)); prefix="Trunc_GS_"*code*"_dim=$trunc_dim"*"_scheme=$trunc_scheme", force=force) do cfg
        return TruncState(cfg, trunc_dim; trunc_scheme=trunc_scheme)
    end
    return data
end


####################
# State properties #
####################

"""
    dim_state(ψ::InfiniteMPS)

Determine the bond dimensions in an infinite MPS.
"""
function dim_state(ψ::InfiniteMPS)
    dimension = Int64.(zeros(length(ψ)))
    for i in 1:length(ψ)
        dimension[i] = dim(space(ψ.AL[i],1))
    end
    return dimension
end

"""
    density_spin(model::Union{OB_Sim,MB_Sim})

Compute the density of spin up and spin down per site in the unit cell for the ground state.
"""
function density_spin(simul::Union{OB_Sim,MB_Sim})
    P = simul.P;
    Q = simul.Q

    dictionary = produce_groundstate(simul);
    ψ₀ = dictionary["groundstate"];
    
    spin::Bool = get(simul.kwargs, :spin, false)

    if !spin
        error("This system is spin independent.")
    end

    return density_spin(ψ₀, P, Q)
end

function density_spin(ψ₀::InfiniteMPS, P::Int64, Q::Int64)
    I, Ps = SymSpace(P,Q,true)
    if iseven(P)
        T = Q
    else 
        T = 2*Q
    end
    Bands = Int(length(ψ₀)/T)

    nup = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(nup)[I((0, 0, 2*Q-P))] .= 1
    blocks(nup)[I((1, 1, Q-P))] .= 1
    ndown = TensorMap(zeros, ComplexF64, Ps ← Ps)
    blocks(ndown)[I((0, 0, 2*Q-P))] .= 1
    blocks(ndown)[I((1, -1, Q-P))] .= 1

    Nup = zeros(Bands,T);
    Ndown = zeros(Bands,T);
    for i in 1:Bands
        for j in 1:T
            Nup[i,j] = real(expectation_value(ψ₀, (i+(j-1)*Bands) => nup))
            Ndown[i,j] = real(expectation_value(ψ₀, (i+(j-1)*Bands) => ndown))
        end
    end

    return Nup, Ndown
end

"""
    density_state(model::Simulation)

Compute the number of electrons per site in the unit cell for the ground state.
"""
function density_state(simul::Union{OB_Sim,MB_Sim})
    P = simul.P;
    Q = simul.Q

    dictionary = produce_groundstate(simul);
    ψ₀ = dictionary["groundstate"];
    
    spin::Bool = get(simul.kwargs, :spin, false)

    return density_state(ψ₀, P, Q, spin)
end

function density_state(simul::Union{OBC_Sim, MBC_Sim})
    dictionary = produce_groundstate(simul);
    ψ = dictionary["groundstate"];

    return density_state(ψ)
end

# For Hubbard models without chemical potential
function density_state(ψ₀::InfiniteMPS,P::Int64,Q::Int64,spin::Bool)
    if iseven(P)
        T = Q
    else 
        T = 2*Q
    end
    Bands = Int(length(ψ₀)/T)

    n = Number(P,Q,spin)

    Nₑ = zeros(Bands*T,1);
    for i in 1:(Bands*T)
        Nₑ[i] = real(expectation_value(ψ₀, i => n))
    end
    
    N_av = zeros(Bands,1)
    for i in 1:Bands
        av = 0
        for j in 0:(T-1)
            av = Nₑ[i+Bands*j] + av
        end
        N_av[i,1] = av/T
    end

    check = (sum(Nₑ)/(T*Bands) ≈ P/Q)
    println("Filling is conserved: $check")

    return Nₑ
end

# For Hubbard models involving a chemical potential
function density_state(ψ::InfiniteMPS)
    Bands = length(ψ)

    n = Number()

    Nₑ = zeros(Bands);
    for i in 1:Bands
        Nₑ[i] = real(expectation_value(ψ, i => n))
    end

    if Bands==1
        # convert 1x1 matrix into scalar
        Nₑ = sum(Nₑ)
    end

    return Nₑ
end


####################
# Tools & Plotting #
####################

"""
    plot_excitations(momenta, energies; title="Excitation_energies", l_margin=[15mm 0mm])

Plot the obtained energy levels in functions of the momentum.
"""
function plot_excitations(momenta, Es; title="Excitation energies", l_margin=[15mm 0mm])
    _, nums = size(Es)
    plot(momenta,real(Es[:,1]), label="", linecolor=:blue, title=title, left_margin=l_margin)
    for i in 2:nums
        plot!(momenta,real(Es[:,i]), label="", linecolor=:blue)
    end
    xlabel!("k")
    ylabel!("Energy density")
end

"""
    plot_spin(model::Simulation; title="Spin Density", l_margin=[15mm 0mm])

Plot the spin density of the model throughout the unit cell as a heatmap.
"""
function plot_spin(model::Simulation; title="Spin Density", l_margin=[15mm 0mm])
    up, down = hf.density_spin(model)
    Sz = up - down
    heatmap(Sz, color=:grays, c=:grays, label="", xlabel="Site", ylabel="Band", title=title, clims=(-1, 1))
end

"""
    extract_params(path::String; range_u::Int64= 1, range_t::Int64=2, range_J::Int64=1, 
                        range_U13::Int64=1, r_1111::Int64 = 1, r_112::Int64 = 1)

Extract the parameters from a params.jl file located at `path` in PyFoldHub format.
"""
function extract_params(path::String; range_u::Int64= 1, range_t::Int64=2, range_J::Int64=1, 
                        range_U13::Int64=1, r_1111::Int64 = 1, r_112::Int64 = 1)
    # Wmn should be rank 8 tensor (only one frequency point)
    include(path)

    B = size(Wmn)[5]
    site_0 = ceil(Int,size(Wmn)[1]/2)

    t = zeros(B,B*range_t)
    U = zeros(B,B*range_u)
    J = zeros(B,B*range_J)
    U13_OS = zeros(B,B)
    if range_U13 == 1
        U13_IS = zeros(B,B*range_U13,4)
    else
        U13_IS = zeros(B,B*(range_U13-1),4)
    end
    for i in 1:B
        for j in 1:B
            for r in 0:(range_t-1)
                t[i,j+r*B] = tmn[site_0+r,i,j] + corr_H[site_0+r,i,j] #+ corr_G_HW[site_0+r,i,j] + corr_v_xc[site_0+r,i,j], check minus sign...
            end
            for r in 0:(range_u-1)
                U[i,j+r*B] = Wmn[site_0,site_0,site_0+r,site_0+r,i,i,j,j]
            end
            for r in 0:(range_J-1)
                if r!=0 || i!=j
                    J[i,j+r*B] = Wmn[site_0,site_0+r,site_0+r,site_0,i,j,j,i]
                    if !(J[i,j+r*B] ≈ Wmn[site_0,site_0+r,site_0,site_0+r,i,j,i,j])
                        error("J1 is not equal to J2 at (r,i,j)=($r,$i,$j).")
                    end
                end
            end
            for r in 1:(range_U13-1)
                U13_IS[i,j+(r-1)*B,1] = Wmn[site_0,site_0+r,site_0+r,site_0+r,i,j,j,j]
                U13_IS[i,j+(r-1)*B,2] = Wmn[site_0+r,site_0+r,site_0,site_0+r,j,j,i,j]
                U13_IS[i,j+(r-1)*B,3] = Wmn[site_0+r,site_0,site_0,site_0,j,i,i,i]
                U13_IS[i,j+(r-1)*B,4] = Wmn[site_0,site_0,site_0+r,site_0,i,i,j,i]
                if !(U13_IS[i,j+(r-1)*B,1] ≈ Wmn[site_0+r,site_0,site_0+r,site_0+r,j,i,j,j]) || !(U13_IS[i,j+(r-1)*B,2] ≈ Wmn[site_0+r,site_0+r,site_0+r,site_0,j,j,j,i]) ||
                    !(U13_IS[i,j+(r-1)*B,3] ≈ Wmn[site_0,site_0+r,site_0,site_0,i,j,i,i]) || !(U13_IS[i,j+(r-1)*B,4] ≈ Wmn[site_0,site_0,site_0,site_0+r,i,i,i,j])
                    error("U13_IS not consistent.")
                end
            end
            if i != j
                U13_OS[i,j] = Wmn[site_0,site_0,site_0,site_0,i,j,j,j]
                if !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,i,j,j], rtol=1e-3) || !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,j,i,j], rtol=1e-3) || 
                    !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,j,j,i], rtol=1e-3)
                    @warn "U13_OS not consistent at i=$i, j=$j, for rtol=1e-3."
                    if !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,i,j,j], atol=1e-3) || !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,j,i,j], atol=1e-3) || 
                        !isapprox(U13_OS[i,j], Wmn[site_0,site_0,site_0,site_0,j,j,j,i], atol=1e-3)
                        error("U13_OS not consistent at i=$i, j=$j.")
                    end
                end
            end
        end
    end

    #shift chemical potential
    mu = minimum(diag(t[:,1:B]))
    t[:,1:B] -= mu.*I

    U112 = Dict{Tuple{Int, Int, Int, Int}, Float64}()
    for i in 1:r_112*B, j in 1:r_112*B, k in 1:r_112*B, l in 1:r_112*B
        if (i == j || i == k || i == l || j == k || j == l || k == l) && length(unique((i, j, k, l))) == 3 && minimum((i,j,k,l)) <= B
            mod_i = mod(i-1,B) + 1; r_i = (i - 1) ÷ B;
            mod_j = mod(j-1,B) + 1; r_j = (j - 1) ÷ B;
            mod_k = mod(k-1,B) + 1; r_k = (k - 1) ÷ B;
            mod_l = mod(l-1,B) + 1; r_l = (l - 1) ÷ B;
            # change index order to those of operators
            U112[(i,k,l,j)] = Wmn[site_0+r_i,site_0+r_j,site_0+r_k,site_0+r_l,mod_i,mod_j,mod_k,mod_l]
        end
    end

    U1111 = Dict{Tuple{Int, Int, Int, Int}, Float64}()
    for i in 1:r_1111*B, j in 1:r_1111*B, k in 1:r_1111*B, l in 1:r_1111*B
        if length(unique((i, j, k, l))) == 4 && minimum((i,j,k,l)) <= B
            mod_i = mod(i-1,B) + 1; r_i = (i - 1) ÷ B;
            mod_j = mod(j-1,B) + 1; r_j = (j - 1) ÷ B;
            mod_k = mod(k-1,B) + 1; r_k = (k - 1) ÷ B;
            mod_l = mod(l-1,B) + 1; r_l = (l - 1) ÷ B;
            # change index order to those of operators
            U1111[(i,k,l,j)] = Wmn[site_0+r_i,site_0+r_j,site_0+r_k,site_0+r_l,mod_i,mod_j,mod_k,mod_l]
        end
    end

    return t, U, J, U13_OS, U13_IS, U112, U1111
end

function save_state(ψ::InfiniteMPS, path::String, name::String)
    path = joinpath(path,name)
    mkdir(path)
    for i in 1:length(ψ)
        d = convert(Dict,ψ.AL[i])
        @save joinpath(path,"state$i.jld2") d
        println("State $i saved.")
    end
end

function load_state(path::String)
    entries = readdir(path)
    file_count = count(entry -> isfile(joinpath(path, entry)), entries)

    @load joinpath(path,"state1.jld2") d
    A = [convert(TensorMap, d)]
    for i in 2:file_count
        @load joinpath(path,"state$i.jld2") d
        push!(A, convert(TensorMap, d))
    end

    return InfiniteMPS(PeriodicArray(A))
end

        
end
