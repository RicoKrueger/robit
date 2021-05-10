function update_v(data::NonHierData, β::Vector{Float64})
    v = zeros(data.J-1, data.N)
    for j in 1:(data.J-1), n in 1:data.N
        v[j,n] = sum(data.x[:,j,n] .* β)
    end
    return v
end

function next_w!(
    data::Data,
    w::Array{Float64,2}, v::Array{Float64,2},
    Σ::Symmetric{Float64,Array{Float64,2}},
    Σ_inv::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    for (j, k) in enumerate(data.nonbase)
        nonj = [jj for jj in 1:(data.J-1) if jj != j]
        Σ_inv_nonj_nonj = inv(Σ[nonj,nonj])
        σ = sqrt(Σ[j,j] - (Σ[j,nonj]' * Σ_inv_nonj_nonj * Σ[nonj,j])[1,1])
        for n in 1:data.N
            μ = v[j,n] + (Σ[j,nonj]' * Σ_inv_nonj_nonj * (w[nonj,n] - v[nonj,n]))[1,1]
            if data.choice[n] == k
                a = maximum(push!(w[nonj,n], 0.0))
                w[j,n] = truncnorm_below_rnd(μ, σ, a)
            elseif data.choice[n] == data.base
                w[j,n] = truncnorm_above_rnd(μ, σ, 0.0)
            else
                choice_id = data.choice[n] - 1 * (data.choice[n] > data.base)
                b = maximum([w[choice_id,n], 0.0])
                w[j,n] = truncnorm_above_rnd(μ, σ, b)
            end
        end
    end
    α = sqrt(tr(Λ * Σ_inv) ./ rand(Chisq(ρ .* (data.J-1))))
    w_trans = α .* w
    return α, w_trans
end

function next_β(
    data::NonHierData,
    w_trans::Array{Float64,2}, Σ_inv::Symmetric{Float64,Array{Float64,2}},
    β_μ0_trans::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    x_Σ_inv_x = zeros(data.K,data.K)
    x_Σ_inv_w = zeros(data.K)
    for n in 1:data.N
        x = data.x[:,:,n]
        x_Σ_inv = x * Σ_inv
        x_Σ_inv_x += x_Σ_inv * x'
        x_Σ_inv_w += x_Σ_inv * w_trans[:,n]
    end

    β_Σ_inv = x_Σ_inv_x + β_B0
    β_μ = β_Σ_inv \ (x_Σ_inv_w + β_B0 * β_μ0_trans)

    v_hat = update_v(data, β_μ)
    z = w_trans - v_hat
    α2 = 0.0
    for n in 1:data.N
        z_n = z[:,n]
        α2 += z_n' * Σ_inv * z_n
    end
    α2 += (β_μ - β_μ0_trans)' * β_B0 * (β_μ - β_μ0_trans) + tr(Λ * Σ_inv)
    α2 /= rand(Chisq((data.N + ρ) .* (data.J-1)))

    β_Σ_inv = Symmetric(β_Σ_inv ./ α2)
    β_trans = mvn_prec_rnd(β_μ, β_Σ_inv)

    v_trans = update_v(data, β_trans)

    return β_trans, v_trans
end

function next_Σ!(
    data::Data,
    w_trans::Array{Float64,2}, v_trans::Array{Float64,2},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    z = w_trans - v_trans
    Σ_trans = Symmetric(rand(InverseWishart(ρ + data.N, Λ .+ z * z')))
    α2 = tr(Σ_trans) ./ (data.J-1)
    α = sqrt(α2)
    Σ = Symmetric(Σ_trans ./ α2)
    Σ_inv = inv(Σ)
    return Σ, Σ_inv, α
end

function mcmc_chain(
    chain_id::Int64, model::Probit, mcmc_args::McmcArgs, data::NonHierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )

    #Initialise storage
    β_store = zeros(data.K,mcmc_args.n_keep)
    Σ_store = zeros(data.J-1, data.J-1,mcmc_args.n_keep)

    #Initialise parameters
    β = randn(data.K)
    Σ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    Σ_inv = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    v = update_v(data, β)
    w = zeros(data.J-1, data.N)

    #Simulate
    j = 0
    for i in 1:mcmc_args.n_iter
        α, w_trans = next_w!(data, w, v, Σ, Σ_inv, ρ, Λ)
        β_μ0_trans = α .* β_μ0
        β_trans, v_trans = next_β(data, w_trans, Σ_inv, β_μ0_trans, β_B0, ρ, Λ)
        Σ, Σ_inv, α = next_Σ!(data, w_trans, v_trans, ρ, Λ)
        β = β_trans ./ α
        w = w_trans ./ α
        v = v_trans ./ α

        if (i > mcmc_args.n_burn) & ((i % mcmc_args.n_thin) == 0)
            j += 1
            β_store[:,j] = β
            Σ_store[:,:,j] = Σ
        end

        print_progress(chain_id, i, mcmc_args)
    end
    return β_store, Σ_store
end

function estimate(
    model::Probit, mcmc_args::McmcArgs, data::NonHierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}}
    )

    ρ = Float64(data.J)
    Λ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))

    println(" ")
    println("Starting MCMC simulation for probit model.")
    println(" ")

    time = @elapsed res = map(c -> mcmc_chain(
            c, model, mcmc_args, data,
            β_μ0, β_B0, ρ, Λ
            ), 1:mcmc_args.n_chain)

    β_store = zeros(data.K, mcmc_args.n_keep, mcmc_args.n_chain)
    Σ_store = zeros(data.J-1, data.J-1, mcmc_args.n_keep, mcmc_args.n_chain)

    for c in 1:mcmc_args.n_chain
        β_store[:,:,c] = res[c][1]
        Σ_store[:,:,:,c] = res[c][2]
    end

    println(" ")
    println("MCMC simulation completed.")
    println(" ")
    println("Estimation time [s]: ", time)
    println(" ")

    summary = Dict(
        "beta" => post_summary(β_store, "beta", data.label_x),
        "Sigma" => post_summary(Σ_store, data.nonbase),
        "time" => time,
        "mcmc_args" => mcmc_args,
        "base" => data.base
        )

    return β_store, Σ_store, summary
end

function remap_w_Σ(
    w::Vector{Float64}, Σ::Symmetric{Float64,Array{Float64,2}},
    nonbase::Vector{Int64}, choice::Int64
    )
    w_new = copy(w)
    Σ_new = Matrix(copy(Σ))
    nonbase_new = copy(nonbase)
    choice_i = findfirst(nonbase .== choice)
    deleteat!(nonbase_new, choice_i)
    pushfirst!(nonbase_new, choice)
    id = collect(1:length(nonbase))
    id_new = [findfirst(nonbase_new .== nonbase[i]) for i in id]

    for (i, i_new) in zip(id, id_new)
        w_new[i_new] = w[i]
    end
    for (i, i_new) in zip(id, id_new), (j, j_new) in zip(id, id_new)
        Σ_new[i_new,j_new] = Σ[i,j]
    end
    Σ_new = Symmetric(Σ_new)
    return w_new, Σ_new
end

function probit_prob_ghk(
    w_in::Vector{Float64}, Σ_in::Symmetric{Float64,Array{Float64,2}},
    base::Int64, nonbase::Vector{Int64}, choice::Int64,
    n_draws::Int64
    )
    if (choice == base) | (findfirst(nonbase .== choice) == 1)
        w = copy(w_in)
        Σ = copy(Σ_in)
    else
        w, Σ = remap_w_Σ(w_in, Σ_in, nonbase, choice)
    end

    Λ = cholesky(Σ).L
    J = length(w) + 1
    η = zeros(J-2)
    p = 0.0
    for d in 1:n_draws
        if choice == base
            μ = -w[1] / Λ[1,1]
            p_d = normcdf(μ)
            η[1] = truncnorm_above_rnd(0.0, 1.0, μ)

            for j in 2:(J-1)
                μ = -(w[j] .+ sum(Λ[j,1:(j-1)] .* η[1:(j-1)])) / Λ[j,j]
                p_d *= normcdf(μ)
                if j < (J - 1)
                    η[j] = truncnorm_above_rnd(0.0, 1.0, μ)
                end
            end
        else
            μ = -w[1] / Λ[1,1]
            p_d = 1.0 - normcdf(μ)
            η[1] = truncnorm_below_rnd(0.0, 1.0, μ)
            u = w[1] + Λ[1,1] .* η[1]

            for j in 2:(J-1)
                μ = (u - (w[j] .+ sum(Λ[j,1:(j-1)] .* η[1:(j-1)]))) / Λ[j,j]
                p_d *= normcdf(μ)
                if j < (J - 1)
                    η[j] = truncnorm_above_rnd(0.0, 1.0, μ)
                end
            end
        end
        p += p_d
    end
    p /= n_draws
    return p
end

function probit_probs_fs(
    w::Vector{Float64}, Σ::Symmetric{Float64,Array{Float64,2}},
    base::Int64, nonbase::Vector{Int64}, choice::Int64,
    n_draws::Int64
    )
    J = length(w) + 1
    Λ = cholesky(Σ).L

    probs = zeros(J)

    for d in 1:n_draws
        eps = Λ * randn(J-1)
        u = w + eps
        if maximum(u) < 0
            choice_rnd = base
        else
            choice_rnd = nonbase[argmax(u)]
        end
        probs[choice_rnd] += 1
    end
    probs ./= n_draws
    return probs
end

function predict_probs(
    data::NonHierData,
    β::Vector{Float64}, Σ::Symmetric{Float64,Array{Float64,2}},
    n_draws::Int64
    )
    #Probabilities
    v = update_v(data, β)
    probs = zeros(data.J, data.N)
    for n in 1:data.N, j in 1:data.J
        probs[j,n] = probit_prob_ghk(v[:,n], Σ, data.base, data.nonbase, j, n_draws)
    end

    #Log-lik
    loglik = 0
    for n in 1:data.N
        loglik += log(probs[data.choice[n],n])
    end
    return probs, loglik
end

function elas(
    data0::NonHierData, probs0::Array{Float64,2},
    df0::DataFrame, label_alt_id::String, label_chosen::String,
    label_x::Vector{String}, base::Int64,
    β::Vector{Float64}, Σ::Symmetric{Float64,Array{Float64,2}},
    n_draws::Int64, j::Int64, x::String, δ::Float64)

    df1 = deepcopy(df0)
    if x == "log_price"
        df1[df1[!,"alt_id"] .== j,"price"] .*= (1 + δ)
        df1[!,"log_price"] = log.(df1[:,"price"])
    else
        df1[df1[!,"alt_id"] .== j,x] .*= (1 + δ)
    end
    data1 = NonHierData(df1, label_alt_id, label_chosen, label_x, base)

    w0 = mean(probs0, dims=2)[:]
    probs1 = predict_probs(data1, β, Σ, n_draws)[1]
    w1 = mean(probs1, dims=2)[:]

    w_δ = (w1 .- w0) ./ w0
    η = w_δ ./ δ
    #w_δ = (w1 .- w0) ./ (w1 .+ w0)
    #x_δ = δ / (2 + δ)
    #η = w_δ ./ x_δ

    Δ = mean(probs1 .- probs0, dims=2)[:]
    return η, Δ, probs1
end
