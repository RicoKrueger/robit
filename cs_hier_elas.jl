push!(LOAD_PATH, ".")
#using Revise

using LdvChoice

using Random, DataFrames, CSV, LinearAlgebra, Statistics, StatsBase, JLD

Random.seed!(1234)

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
df_test = df[df[:,"train_set"] .== 0,:]

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

hier_data_train = HierData(
    df_train,
    label_ind_id, label_obs_id, label_alt_id,
    label_chosen, label_x_fix, label_x_rnd, base)
hier_data_test = HierData(
    df_test,
    label_ind_id, label_obs_id, label_alt_id,
    label_chosen, label_x_fix, label_x_rnd, base)

###
#Load results
###

models = ["hier_probit", "hier_robit", "hier_genrobit"]

results = Dict()
β_mean = Dict()
η_mean = Dict()
ω_mean = Dict()
Σ_mean = Dict()
ν_mean = Dict()
for m in models
    results[m] = load("results_$(m)_cs_base$(base).jld")
    β_mean[m] = mean(results[m]["beta_store"], dims=(2,3))[:]
    Σ_mean[m] = Symmetric(mean(results[m]["Sigma_store"], dims=(3,4))[:,:])
end

ν_mean["hier_robit"] =  mean(results["hier_robit"]["nu_store"])
ν_mean["hier_genrobit"] =  mean(results["hier_genrobit"]["nu_store"], dims=(2,3))[:]

η_mean["hier_probit"] = mean(results["hier_probit"]["eta_store"], dims=(2,3))[:]
ω_mean["hier_probit"] = mean(results["hier_probit"]["omega_store"], dims=(2,3))[:]

η_mean["hier_robit"] = mean(results["hier_robit"]["eta_store"], dims=(2,3))[:]
ω_mean["hier_robit"] = mean(results["hier_robit"]["omega_store"], dims=(2,3))[:]

η_mean["hier_genrobit"] = mean(results["hier_genrobit"]["eta_store"], dims=(2,3))[:]
ω_mean["hier_genrobit"] = mean(results["hier_genrobit"]["omega_store"], dims=(2,3))[:]

###
#Prediction: In-sample
###

n_draws = Dict("hier_probit" => 200, "hier_robit" => 500, "hier_genrobit" => 500)
n_sim = 200

probs = Dict()
loglik = Dict()

probs["hier_probit"], loglik["hier_probit"] = predict_probs(
        hier_data_train,
        β_mean["hier_probit"], η_mean["hier_probit"], ω_mean["hier_probit"],
        Σ_mean["hier_probit"],
        n_sim, n_draws["hier_probit"]
        )
probs["hier_robit"], loglik["hier_robit"] = predict_probs(
        hier_data_train,
        β_mean["hier_robit"], η_mean["hier_robit"], ω_mean["hier_robit"],
        Σ_mean["hier_robit"], ν_mean["hier_robit"],
        n_sim, n_draws["hier_robit"]
        )
probs["hier_genrobit"], loglik["hier_genrobit"] = predict_probs(
        hier_data_train,
        β_mean["hier_genrobit"], η_mean["hier_genrobit"], ω_mean["hier_genrobit"],
        Σ_mean["hier_genrobit"], ν_mean["hier_genrobit"],
        n_sim, n_draws["hier_genrobit"]
        )

###
#Prediction: Out-of-sample
###

probs_test = Dict()
loglik_test = Dict()

probs_test["hier_probit"], loglik_test["hier_probit"] = predict_probs(
        hier_data_test,
        β_mean["hier_probit"], η_mean["hier_probit"], ω_mean["hier_probit"],
        Σ_mean["hier_probit"],
        n_sim, n_draws["hier_probit"]
        )
probs_test["hier_robit"], loglik_test["hier_robit"] = predict_probs(
        hier_data_test,
        β_mean["hier_robit"], η_mean["hier_robit"], ω_mean["hier_robit"],
        Σ_mean["hier_robit"], ν_mean["hier_robit"],
        n_sim, n_draws["hier_robit"]
        )
probs_test["hier_genrobit"], loglik_test["hier_genrobit"] = predict_probs(
        hier_data_test,
        β_mean["hier_genrobit"], η_mean["hier_genrobit"], ω_mean["hier_genrobit"],
        Σ_mean["hier_genrobit"], ν_mean["hier_genrobit"],
        n_sim, n_draws["hier_genrobit"]
        )

###
#Scores
###

function brier_score(data::Data, probs::Array{Float64,2})
    bs = 0.0
    for n in 1:data.N, j in 1:data.J
        bs += ((data.choice[n] == j) - probs[j,n]).^2
    end
    return bs
end

function log_score(data::Data, probs::Array{Float64,2})
    ls_n = [log(probs[data.choice[n],n]) for n in 1:data.N]
    ls = sum(ls_n)
    return ls, ls_n
end

function akaike_ic(loglik::Float64, k::Int64)
    aic = - 2 .* loglik .+ 2 .* k
    return aic
end

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))

bs = [brier_score(hier_data_train, probs[m]) for m in models]
bs_test = [brier_score(hier_data_test, probs_test[m]) for m in models]
#ls, ls_n = unzip([log_score(hier_data_train, probs[m]) for m in models])
#ls_test, ls_n_test = unzip([log_score(hier_data_test, probs_test[m]) for m in models])

function dict_to_vec(dict::Dict, keys::Vector{String})
    return [dict[k] for k in keys]
end

params = Dict()
params["hier_probit"] = size(β_mean["hier_probit"])[1] +
    2 * size(η_mean["hier_probit"])[1] +
    Int(hier_data_train.J .* (hier_data_train.J - 1) / 2)
params["hier_robit"] = params["hier_probit"] + 1
params["hier_genrobit"] = params["hier_probit"] + hier_data_train.J - 1

df_fit = DataFrame(models = models)
df_fit."params" = dict_to_vec(params, models)
df_fit."loglik" = dict_to_vec(loglik, models)
df_fit."aic" = - 2 .* df_fit."loglik" .+ 2 .* df_fit."params"
df_fit."loglik_test" = dict_to_vec(loglik_test, models)
df_fit."aic_test" = - 2 .* df_fit."loglik_test" .+ 2 .* df_fit."params"
df_fit."bs" = bs
df_fit."bs_test" = bs_test

println(df_fit)

CSV.write("fit_cs_hier_base$(base).csv", df_fit)

#=
###
#Elasticities
###

cols = ["alt$(j)" for j in 1:hier_data_train.J]
df_η = Dict(m => DataFrame(Dict(c => [] for c in cols)) for m in models)

function elas!(j::Int64, x::String, δ_vec::Vector{Float64}, models::Vector{String})
    for m in models, δ in δ_vec
        if m == "hier_probit"
            η = elas(
                hier_data_train, probs[m],
                df,
                label_ind_id, label_obs_id, label_alt_id, label_chosen,
                label_x_fix, label_x_rnd,
                base,
                β_mean[m], η_mean[m], ω_mean[m],
                Σ_mean[m],
                n_sim, n_draws[m], j, x, δ
                )
        else
            η = elas(
                hier_data_train, probs[m],
                df,
                label_ind_id, label_obs_id, label_alt_id, label_chosen,
                label_x_fix, label_x_rnd,
                base,
                β_mean[m], η_mean[m], ω_mean[m],
                Σ_mean[m], ν_mean[m],
                n_sim, n_draws[m], j, x, δ
                )
        end
        df_η[m] = push!(df_η[m], η)
    end
end

#Driving cost
j = 4
x = "cost"
δ_vec = [0.05, 0.10, 0.25]
elas!(j, x, δ_vec, models)

#Driving in-vehicle travel time
j = 4
x = "ivtt"
δ_vec = [0.05, 0.10, 0.25]
elas!(j, x, δ_vec, models)

#Driving travel time uncertainty
j = 4
x = "traffic_var"
δ_vec = [-0.05, -0.10, -0.25]
elas!(j, x, δ_vec, models)

#PT fares
j = 3
x = "cost"
δ_vec = [0.05, 0.10, 0.25]
elas!(j, x, δ_vec, models)

#PT in-vehicle travel time decreased by 5%, 10%, 25%
j = 3
x = "ivtt"
δ_vec = [-0.05, -0.10, -0.25]
elas!(j, x, δ_vec, models)

#PT out-of-vehicle travel time
j = 3
x = "ovtt"
δ_vec = [-0.05, -0.10, -0.25]
elas!(j, x, δ_vec, models)

#Cycling travel time
j = 2
x = "ovtt"
δ_vec = [-0.05, -0.10, -0.25]
elas!(j, x, δ_vec, models)

#Walking travel time
j = 1
x = "ovtt"
δ_vec = [-0.05, -0.10, -0.25]
elas!(j, x, δ_vec, models)

for m in models
    CSV.write("elas_cs_hier_$(m).csv", df_η[m])
end
=#
