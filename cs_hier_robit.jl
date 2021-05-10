push!(LOAD_PATH, ".")
#using Revise

using LdvChoice

using DataFrames, CSV, LinearAlgebra, Statistics, StatsBase, JLD

###
#Load and prepare data
###

df = DataFrame(CSV.File("london.csv"))

df[!,"asc2"] = 1 * (df[:,"alt_id"] .== 2)
df[!,"asc3"] = 1 * (df[:,"alt_id"] .== 3)
df[!,"asc4"] = 1 * (df[:,"alt_id"] .== 4)

df[!,"age_youth"] = 1 * (df[:,"age"] .< 18)
df[!,"age_senior"] = 1 * (df[:,"age"] .>= 65)

df[!,"bike_female"] = df[!,"asc2"] .* df[!,"female"]
df[!,"bike_winter"] = df[!,"asc2"] .* df[!,"winter"]

df[!,"pt_female"] = df[!,"asc3"] .* df[!,"female"]
df[!,"pt_age_youth"] = df[!,"asc3"] .* df[!,"age_youth"]
df[!,"pt_age_senior"] = df[!,"asc3"] .* df[!,"age_senior"]

df[!,"drive_female"] = df[!,"asc4"] .* df[!,"female"]
df[!,"drive_age_youth"] = df[!,"asc4"] .* df[!,"age_youth"]
df[!,"drive_age_senior"] = df[!,"asc4"] .* df[!,"age_senior"]

df_train = df[df[:,"train_set"] .== 1,:]

label_ind_id = "ind_id"
label_obs_id = "obs_id"
label_alt_id = "alt_id"
label_chosen = "chosen"
label_x_fix = [
    "asc2", "asc3", "asc4",
    "cost", "ovtt", "transfers",
    "bike_female", "bike_winter",
    "pt_female", "pt_age_youth", "pt_age_senior",
    "drive_female", "drive_age_youth", "drive_age_senior", "car_ownership"]
label_x_rnd = ["ivtt", "traffic_var"]
base = 4

data = HierData(
    df_train,
    label_ind_id, label_obs_id, label_alt_id, label_chosen,
    label_x_fix, label_x_rnd, base)

###
#MCMC simulation
###

model = HierRobit("cs")
mcmc_args = McmcArgs(n_chain=1, n_burn=50000, n_sample=50000, n_thin=10, disp=1000)

β_μ0 = zeros(data.K_fix)
β_B0 = Symmetric(Matrix(1e-6 .* I, data.K_fix, data.K_fix))
C0_inv = Symmetric(Matrix(1e-6 .* I, data.K_rnd, data.K_rnd))
α0 = 1.0
β0 = 0.1

β_store, η_store, ω_store, Σ_store, ν_store, summary =
estimate(model, mcmc_args, data, β_μ0, β_B0, C0_inv, α0, β0)

results = Dict(
    "beta_store" => β_store,
    "eta_store" => η_store,
    "omega_store" => ω_store,
    "Sigma_store" => Σ_store,
    "nu_store" => ν_store,
    "summary" => summary
    )

save("results_$(model.name)_base$(base).jld", results)
