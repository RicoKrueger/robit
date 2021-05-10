module LdvChoice
export Probit, Robit, GenRobit
export HierProbit, HierRobit, HierGenRobit
export Data, NonHierData, HierData, McmcArgs
export estimate, predict_probs
export cov_to_cor, elas

using Truncnorm

using Distributed, DataFrames, LinearAlgebra, Distributions, SpecialFunctions
using Roots

###
#Type definitions
###

include("LdvChoice_types.jl")

###
#Convenience
###

function normcdf(x::Float64)
    return 0.5 * erfc(-x / sqrt(2))
end

function cov_to_cor(Σ::Symmetric{Float64,Array{Float64,2}})
    D = Diagonal(1 ./ sqrt.(diag(Σ)))
    Ω = D * Σ * D
    return Ω
end

function mvn_prec_rnd(
    μ::Vector{Float64},
    Ψ::Symmetric{Float64,Array{Float64,2}}
    )
    return μ + cholesky(Ψ).U \ randn(length(μ))
end

function print_progress(chain_id::Int64, i::Int64, mcmc_args::McmcArgs)
    if (i % mcmc_args.disp) == 0
        if i <= mcmc_args.n_burn
            sample_state = "burn in"
        else
            sample_state = "sampling"
        end
        flush(stdout)
        println("Chain $(chain_id): Iteration $(i) ($(sample_state))")
    end
end

###
#Probit
###

include("LdvChoice_probit.jl")
include("LdvChoice_hier_probit.jl")

###
#Robit
###

include("LdvChoice_robit.jl")
include("LdvChoice_hier_robit.jl")

###
#Generalised robit
###

include("LdvChoice_genrobit.jl")
include("LdvChoice_hier_genrobit.jl")

###
#Post estimation
###

include("LdvChoice_post_estimation.jl")

end
