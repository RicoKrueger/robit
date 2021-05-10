push!(LOAD_PATH, ".")
#using Revise
using LdvChoice

using DataFrames, CSV, LinearAlgebra, Statistics, StatsBase, JLD

###
#Load and prepare data
###

df = DataFrame(CSV.File("synthetic_data_robit.csv"))

df[!,"asc1"] = 1 * (df[:,"alt_id"] .== 1)
df[!,"asc2"] = 1 * (df[:,"alt_id"] .== 2)
df[!,"asc3"] = 1 * (df[:,"alt_id"] .== 3)

label_alt_id = "alt_id"
label_chosen = "chosen"
label_x = ["asc1", "asc2", "asc3", "x1", "x2"]
base = 4

data = NonHierData(df, label_alt_id, label_chosen, label_x, base)

###
#MCMC simulation
###

model = Robit("sim1")
mcmc_args = McmcArgs(n_chain=1, n_burn=50000, n_sample=50000, n_thin=10, disp=1000)

β_μ0 = zeros(data.K)
β_B0 = Symmetric(Matrix(1e-6 .* I, data.K, data.K))
α0 = 1.0
β0 = 0.1

β_store, Σ_store, ν_store, summary = estimate(
    model, mcmc_args, data, β_μ0, β_B0, α0, β0
    )

results = Dict(
    "beta_store" => β_store,
    "Sigma_store" => Σ_store,
    "nu_store" => ν_store,
    "summary" => summary
    )

save("results_$(model.name).jld", results)
