 function update_v_fix(data::HierData, β::Vector{Float64})
    v_fix = zeros(data.J-1, data.N)
    for j in 1:(data.J-1), n in 1:data.N
        v_fix[j,n] = sum(data.x_fix[:,j,n] .* β)
    end
    return v_fix
end

function update_v_rnd(data::HierData, γ::Array{Float64,2})
    v_rnd = zeros(data.J-1, data.N)
    for j in 1:(data.J-1), n in 1:data.N
        v_rnd[j,n] = sum(data.x_rnd[:,j,n] .* γ[:,data.obs_to_ind[n]])
    end
    return v_rnd
end

function next_β(
    data::HierData,
    w_trans::Array{Float64,2}, Σ_inv::Symmetric{Float64,Array{Float64,2}},
    β_μ0_trans::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    v_rnd_trans::Array{Float64,2},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    z_trans = w_trans .- v_rnd_trans
    x_Σ_inv_x = zeros(data.K_fix,data.K_fix)
    x_Σ_inv_z = zeros(data.K_fix)
    for n in 1:data.N
        x = data.x_fix[:,:,n]
        x_Σ_inv = x * Σ_inv
        x_Σ_inv_x .+= x_Σ_inv * x'
        x_Σ_inv_z .+= x_Σ_inv * z_trans[:,n]
    end

    β_Σ_inv = x_Σ_inv_x + β_B0
    β_μ = β_Σ_inv \ (x_Σ_inv_z + β_B0 * β_μ0_trans)

    v_fix_hat = update_v_fix(data, β_μ)
    z_trans .-= v_fix_hat
    α2 = 0.0
    for n in 1:data.N
        z_n = z_trans[:,n]
        α2 += z_n' * Σ_inv * z_n
    end
    α2 += (β_μ - β_μ0_trans)' * β_B0 * (β_μ - β_μ0_trans) + tr(Λ * Σ_inv)
    α2 /= rand(Chisq((data.N + ρ) .* (data.J-1)))

    β_Σ_inv = Symmetric(β_Σ_inv ./ α2)
    β_trans = mvn_prec_rnd(β_μ, β_Σ_inv)

    v_fix_trans = update_v_fix(data, β_trans)

    return β_trans, v_fix_trans
end

function next_γ(
    data::HierData,
    w_trans::Array{Float64,2}, Σ_inv::Symmetric{Float64,Array{Float64,2}},
    η_trans::Vector{Float64}, Ω_inv::Symmetric{Float64,Array{Float64,2}},
    v_fix_trans::Array{Float64,2},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    z_trans = w_trans .- v_fix_trans
    γ_μ = zeros(data.K_rnd,data.n_ind)
    γ_Σ_inv = zeros(data.K_rnd,data.K_rnd,data.n_ind)
    n = 0
    for i in 1:data.n_ind
        x_Σ_inv_x = zeros(data.K_rnd,data.K_rnd)
        x_Σ_inv_z = zeros(data.K_rnd)
        for t in 1:data.obs_per_ind[i]
            n += 1
            x = data.x_rnd[:,:,n]
            x_Σ_inv = x * Σ_inv
            x_Σ_inv_x .+= x_Σ_inv * x'
            x_Σ_inv_z .+= x_Σ_inv * z_trans[:,n]
        end
        γ_Σ_inv_i = x_Σ_inv_x + Ω_inv
        γ_Σ_inv[:,:,i] = γ_Σ_inv_i
        γ_μ[:,i] = γ_Σ_inv_i \ (x_Σ_inv_z + Ω_inv * η_trans)
    end

    v_rnd_hat = update_v_rnd(data, γ_μ)
    z_trans .-= v_rnd_hat
    α2 = 0.0
    for n in 1:data.N
        z_n = z_trans[:,n]
        α2 += z_n' * Σ_inv * z_n
    end
    for i in 1:data.n_ind
        z_γ_i = γ_μ[:,i] - η_trans
        α2 += z_γ_i' * Ω_inv * z_γ_i
    end
    α2 += tr(Λ * Σ_inv)
    α2 /= rand(Chisq((data.N + ρ) .* (data.J-1)))

    γ_trans = zeros(data.K_rnd,data.n_ind)
    for i in 1:data.n_ind
        γ_Σ_inv_i = Symmetric(γ_Σ_inv[:,:,i] ./ α2)
        γ_trans[:,i] = mvn_prec_rnd(γ_μ[:,i], γ_Σ_inv_i)
    end

    α = sqrt(α2)
    γ = γ_trans ./ α
    v_rnd_trans = update_v_rnd(data, γ_trans)

    return γ, v_rnd_trans
end

function next_η(
    data::HierData,
    γ::Array{Float64,2}, Ω_inv::Symmetric{Float64,Array{Float64,2}},
    C0_inv::Symmetric{Float64,Array{Float64,2}}
    )
    η_Σ_inv = Symmetric(data.n_ind .* Ω_inv .+ C0_inv)
    η_μ = η_Σ_inv \ (Ω_inv * sum(γ, dims=2)[:])
    η = mvn_prec_rnd(η_μ, η_Σ_inv)
    return η
end

function next_ω(
    data::HierData,
    γ::Array{Float64,2}, η::Vector{Float64}, λ::Vector{Float64},
    κ::Float64)
    a = (κ + data.n_ind) / 2
    ω2_inv = [rand(Gamma(a, 1 / (κ * λ[k] + 0.5 * sum((γ[k,:] .- η[k]).^2))))
        for k in 1:data.K_rnd]
    ω2 = 1 ./ ω2_inv
    ω = sqrt.(ω2)
    Ω_inv = Symmetric(diagm(ω2_inv))
    return ω, ω2, Ω_inv
end

function next_λ(
    data::HierData,
    ω2::Vector{Float64}, κ::Float64, δ::Float64
    )
    a = (κ + 1) / 2
    λ = [rand(Gamma(a, 1 / (δ^(-2) + κ / ω2[k]))) for k in 1:data.K_rnd]
    return λ
end

function mcmc_chain(
    chain_id::Int64, model::HierProbit, mcmc_args::McmcArgs, data::HierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    C0_inv::Symmetric{Float64,Array{Float64,2}},
    κ::Float64, δ::Float64,
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )

    #Initialise storage
    β_store = zeros(data.K_fix,mcmc_args.n_keep)
    η_store = zeros(data.K_rnd,mcmc_args.n_keep)
    ω_store = zeros(data.K_rnd,mcmc_args.n_keep)
    Σ_store = zeros(data.J-1, data.J-1,mcmc_args.n_keep)

    #Initialise parameters
    β = randn(data.K_fix)
    γ = randn(data.K_rnd, data.n_ind)
    η = randn(data.K_rnd)
    ω2 = ones(data.K_rnd)
    Ω = Symmetric(diagm(ω2))
    Ω_inv = Symmetric(diagm(1 ./ ω2))
    λ = rand(Gamma(1 / 2, δ^2), data.K_rnd)
    Σ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    Σ_inv = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    α = 1.0
    v_fix = update_v_fix(data, β)
    v_rnd = update_v_rnd(data, γ)
    v = v_fix .+ v_rnd
    w = zeros(data.J-1, data.N)

    #Simulate
    j = 0
    for i in 1:mcmc_args.n_iter
        α, w_trans = next_w!(data, w, v, Σ, Σ_inv, ρ, Λ)
        γ_trans = α .* γ
        η_trans = α .* η
        β_μ0_trans = α .* β_μ0
        v_rnd_trans = update_v_rnd(data, γ_trans)
        β_trans, v_fix_trans = next_β(
            data, w_trans, Σ_inv, β_μ0_trans, β_B0, v_rnd_trans, ρ, Λ
            )
        γ, v_rnd_trans = next_γ(
            data, w_trans, Σ_inv, η_trans, Ω_inv, v_fix_trans, ρ, Λ
            )
        v_trans = v_fix_trans .+ v_rnd_trans
        η = next_η(data, γ, Ω_inv, C0_inv)
        ω, ω2, Ω_inv = next_ω(data, γ, η, λ, κ)
        λ = next_λ(data, ω2, κ, δ)
        Σ, Σ_inv, α = next_Σ!(data, w_trans, v_trans, ρ, Λ)
        β = β_trans ./ α
        w = w_trans ./ α
        v = v_trans ./ α

        if (i > mcmc_args.n_burn) & ((i % mcmc_args.n_thin) == 0)
            j += 1
            β_store[:,j] = β
            η_store[:,j] = η
            ω_store[:,j] = ω
            Σ_store[:,:,j] = Σ
        end

        print_progress(chain_id, i, mcmc_args)
    end
    return β_store, η_store, ω_store, Σ_store
end

function estimate(
    model::HierProbit, mcmc_args::McmcArgs, data::HierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    C0_inv::Symmetric{Float64,Array{Float64,2}}
    )

    κ = 2.0
    δ = 1e3

    ρ = Float64(data.J)
    Λ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))

    println(" ")
    println("Starting MCMC simulation for hierarchical probit model.")
    println(" ")

    time = @elapsed res = map(c -> mcmc_chain(
            c, model, mcmc_args, data,
            β_μ0, β_B0, C0_inv, κ, δ, ρ, Λ,
            ), 1:mcmc_args.n_chain)

    β_store = zeros(data.K_fix, mcmc_args.n_keep, mcmc_args.n_chain)
    η_store = zeros(data.K_rnd, mcmc_args.n_keep, mcmc_args.n_chain)
    ω_store = zeros(data.K_rnd, mcmc_args.n_keep, mcmc_args.n_chain)
    Σ_store = zeros(data.J-1, data.J-1, mcmc_args.n_keep, mcmc_args.n_chain)

    for c in 1:mcmc_args.n_chain
        β_store[:,:,c] = res[c][1]
        η_store[:,:,c] = res[c][2]
        ω_store[:,:,c] = res[c][3]
        Σ_store[:,:,:,c] = res[c][4]
    end

    println(" ")
    println("MCMC simulation completed.")
    println(" ")
    println("Estimation time [s]: ", time)
    println(" ")

    summary = Dict(
        "beta" => post_summary(β_store, "beta", data.label_x_fix),
        "eta" => post_summary(η_store, "eta", data.label_x_rnd),
        "omega" => post_summary(ω_store, "omega", data.label_x_rnd),
        "Sigma" => post_summary(Σ_store, data.nonbase),
        "time" => time,
        "mcmc_args" => mcmc_args,
        "base" => data.base
        )

    return β_store, η_store, ω_store, Σ_store, summary
end

function predict_probs(
    data::HierData,
    β::Vector{Float64}, η::Vector{Float64}, ω::Vector{Float64},
    Σ::Symmetric{Float64,Array{Float64,2}},
    n_sim::Int64, n_draws::Int64
    )
    #Probabilities
    v_fix = update_v_fix(data, β)
    probs_sim = zeros(n_sim, data.J, data.N)
    for r in 1:n_sim
        γ = repeat(η, 1, data.n_ind) .+
            repeat(ω, 1, data.n_ind) .* randn(data.K_rnd, data.n_ind)
        v_rnd = update_v_rnd(data, γ)
        v = v_fix .+ v_rnd
        for n in 1:data.N, j in 1:data.J
            probs_sim[r,j,n] += probit_prob_ghk(v[:,n], Σ, data.base,
                data.nonbase, j, n_draws)
        end
    end
    probs = mean(probs_sim, dims=1)[1,:,:]

    #Log-lik
    log_probs_ind_sim = zeros(n_sim, data.n_ind)
    n = 0
    for i in 1:data.n_ind, t in 1:data.obs_per_ind[i]
        n += 1
        log_probs_ind_sim[:,i] += log.(probs_sim[:,data.choice[n],n])
    end
    probs_ind = mean(exp.(log_probs_ind_sim), dims=1)[:]
    loglik = sum(log.(probs_ind))
    return probs, loglik
end

function elas(
    data0::HierData, probs0::Array{Float64,2},
    df0::DataFrame,
    label_ind_id::String, label_obs_id::String,
    label_alt_id::String, label_chosen::String,
    label_x_fix::Vector{String}, label_x_rnd::Vector{String},
    base::Int64,
    β::Vector{Float64}, η::Vector{Float64}, ω::Vector{Float64},
    Σ::Symmetric{Float64,Array{Float64,2}},
    n_sim::Int64, n_draws::Int64, j::Int64, x::String, δ::Float64
    )
    df1 = deepcopy(df0)
    df1[df1[!,"alt_id"] .== j,x] .*= (1 + δ)
    data1 = HierData(df1, label_ind_id, label_obs_id, label_alt_id, label_chosen,
        label_x_fix, label_x_rnd, base)

    w0 = mean(probs0, dims=2)[:]
    probs1 = predict_probs(data1, β, η, ω, Σ, n_sim, n_draws)[1]
    w1 = mean(probs1, dims=2)[:]

    #w_δ = (w1 .- w0) ./ w0
    #η = w_δ ./ δ
    w_δ = (w1 .- w0) ./ (w1 .+ w0)
    x_δ = δ / (2 + δ)
    η = w_δ ./ x_δ
    return η
end
