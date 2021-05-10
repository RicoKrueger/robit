function q_lpdf(q::Float64, u::Float64, c::Float64, ν_j::Float64)
    return -q / 2 * u - sqrt(q) * c + (ν_j - 1) / 2 * log(q)
end

function q_lprior(q::Float64, α::Float64, β::Float64)
    return (α - 1) * log(q) - β * q
end

function next_q!(
    data::Data,
    w::Array{Float64,2}, v::Array{Float64,2},
    Σ_inv::Symmetric{Float64,Array{Float64,2}}, q::Array{Float64,2},
    ν::Vector{Float64}
    )
    z = w - v
    for n in 1:data.N, j in 1:(data.J-1)
        nonj = [jj for jj in 1:(data.J-1) if jj != j]
        u = ν[j] + Σ_inv[j,j] * z[j,n]^2
        c = z[j,n] .* sum(sqrt.(q[nonj,n]) .* Σ_inv[nonj,j] .* z[nonj,n])

        if ν[j] > 1
            m = ((c / 2 + sqrt((c / 2)^2 + u * (ν[j] - 1))) / (ν[j] - 1))^(-2)
            l = c / 4 / sqrt(m^3) - (ν[j] - 1) / 2 / m^2
            α_star = 1 - m^2 * l
            β_star = -m * l
        else
            α_star = 1.0
            β_star = u / 2
        end

        q_new = rand(Gamma(α_star, 1 / β_star))
        log_α = (q_lpdf(q_new, u, c, ν[j]) - q_lprior(q_new, α_star, β_star)
                 - q_lpdf(q[j,n], u, c, ν[j]) + q_lprior(q[j,n], α_star, β_star))
        if log(rand()) <= log_α
            q[j,n] = q_new
        end
    end
end

function next_ν(
    data::Data,
    q::Array{Float64,2}, ν::Vector{Float64},
    α0::Float64, β0::Float64
    )
    ν_new = [next_ν(data, q[j,:], ν[j], α0, β0) for j in 1:(data.J-1)]
    return ν_new
end

function next_w!(
    data::Data,
    w::Array{Float64,2}, v::Array{Float64,2}, q::Array{Float64,2},
    Σ::Symmetric{Float64,Array{Float64,2}},
    Σ_inv::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    for (j, k) in enumerate(data.nonbase)
        nonj = [jj for jj in 1:(data.J-1) if jj != j]
        E = (Σ[j,nonj]' * inv(Σ[nonj,nonj]))[:]
        F = E' * Σ[j,nonj]
        for n in 1:data.N
            σ = sqrt((Σ[j,j] - F) / q[j,n])
            μ = v[j,n] + sum(E .* (w[nonj,n] - v[nonj,n]) .* sqrt.(q[nonj,n])) / sqrt(q[j,n])
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
    w_trans::Array{Float64,2}, q::Array{Float64,2},
    Σ_inv::Symmetric{Float64,Array{Float64,2}},
    β_μ0_trans::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    x_q_Σ_inv_q_x = zeros(data.K,data.K)
    x_q_Σ_inv_q_w = zeros(data.K)
    for n in 1:data.N
        x = data.x[:,:,n]
        Q_ch = Diagonal(sqrt.(q[:,n]))
        x_q_Σ_inv_q = x * Q_ch * Σ_inv * Q_ch
        x_q_Σ_inv_q_x += x_q_Σ_inv_q * x'
        x_q_Σ_inv_q_w += x_q_Σ_inv_q * w_trans[:,n]
    end

    β_Σ_inv = x_q_Σ_inv_q_x + β_B0
    β_μ = β_Σ_inv \ (x_q_Σ_inv_q_w + β_B0 * β_μ0_trans)

    v_hat = update_v(data, β_μ)
    z = w_trans - v_hat
    α2 = 0.0
    for n in 1:data.N
        z_n = z[:,n]
        Q_ch = Diagonal(sqrt.(q[:,n]))
        α2 += (z_n' * Q_ch * Σ_inv * Q_ch * z_n)
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
    w_trans::Array{Float64,2}, q::Array{Float64,2}, v_trans::Array{Float64,2},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}}
    )
    z = w_trans - v_trans
    qzqz = zeros(data.J-1,data.J-1)
    for n in 1:data.N
        qz = Diagonal(sqrt.(q[:,n])) * z[:,n]
        qzqz .+= qz * qz'
    end
    Σ_trans = Symmetric(rand(InverseWishart(ρ + data.N, Λ .+ qzqz)))
    α2 = tr(Σ_trans) ./ (data.J-1)
    α = sqrt(α2)
    Σ = Symmetric(Σ_trans ./ α2)
    Σ_inv = inv(Σ)
    return Σ, Σ_inv, α
end

function mcmc_chain(
    chain_id::Int64, model::GenRobit, mcmc_args::McmcArgs, data::NonHierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    ρ::Float64, Λ::Symmetric{Float64,Array{Float64,2}},
    α0::Float64, β0::Float64
    )

    #Initialise storage
    β_store = zeros(data.K,mcmc_args.n_keep)
    Σ_store = zeros(data.J-1, data.J-1,mcmc_args.n_keep)
    ν_store = zeros(data.J-1, mcmc_args.n_keep)

    #Initialise parameters
    β = randn(data.K)
    Σ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    Σ_inv = Symmetric(Matrix(1.0I, data.J-1, data.J-1))
    α = 1.0
    v = update_v(data, β)
    w = zeros(data.J-1, data.N)
    q = ones(data.J-1, data.N)
    ν = 1 .+ 4 .* rand(data.J-1)

    #Simulate
    j = 0
    for i in 1:mcmc_args.n_iter
        next_q!(data, w, v, Σ_inv, q, ν)
        ν = next_ν(data, q, ν, α0, β0)
        α, w_trans = next_w!(data, w, v, q, Σ, Σ_inv, ρ, Λ)
        β_μ0_trans = α .* β_μ0
        β_trans, v_trans = next_β(
            data, w_trans, q, Σ_inv, β_μ0_trans, β_B0, ρ, Λ
            )
        Σ, Σ_inv, α = next_Σ!(data, w_trans, q, v_trans, ρ, Λ)
        β = β_trans ./ α
        w = w_trans ./ α
        v = v_trans ./ α

        if (i > mcmc_args.n_burn) & ((i % mcmc_args.n_thin) == 0)
            j += 1
            β_store[:,j] = β
            Σ_store[:,:,j] = Σ
            ν_store[:,j] = ν
        end

        print_progress(chain_id, i, mcmc_args)
    end
    return β_store, Σ_store, ν_store
end

function estimate(
    model::GenRobit, mcmc_args::McmcArgs, data::NonHierData,
    β_μ0::Vector{Float64}, β_B0::Symmetric{Float64,Array{Float64,2}},
    α0::Float64, β0::Float64
    )

    ρ = Float64(data.J)
    Λ = Symmetric(Matrix(1.0I, data.J-1, data.J-1))

    println(" ")
    println("Starting MCMC simulation for Gen-robit model.")
    println(" ")

    time = @elapsed res = map(c -> mcmc_chain(
            c, model, mcmc_args, data,
            β_μ0, β_B0, ρ, Λ, α0, β0
            ), 1:mcmc_args.n_chain)

    β_store = zeros(data.K, mcmc_args.n_keep, mcmc_args.n_chain)
    Σ_store = zeros(data.J-1, data.J-1, mcmc_args.n_keep, mcmc_args.n_chain)
    ν_store = zeros(data.J-1, mcmc_args.n_keep, mcmc_args.n_chain)

    for c in 1:mcmc_args.n_chain
        β_store[:,:,c] = res[c][1]
        Σ_store[:,:,:,c] = res[c][2]
        ν_store[:,:,c] = res[c][3]
    end

    println(" ")
    println("MCMC simulation completed.")
    println(" ")
    println("Estimation time [s]: ", time)
    println(" ")

    summary = Dict(
        "beta" => post_summary(β_store, "beta", data.label_x),
        "Sigma" => post_summary(Σ_store, data.nonbase),
        "nu" => post_summary(ν_store, "nu", ["alt$(i)" for i in data.nonbase])
    )

    return β_store, Σ_store, ν_store, summary
end

function genrobit_probs_fs(
    v::Vector{Float64}, Σ_L::LowerTriangular{Float64,Array{Float64,2}}, ν::Vector{Float64},
    J::Int64, base::Int64, nonbase::Vector{Int64}, n_draws::Int64
    )
    probs = zeros(J)
    for d in 1:n_draws
        q = [rand(Chisq(ν[j])) / ν[j] for j in 1:(J-1)]
        Q_ch_inv = Diagonal(1 ./ sqrt.(q))
        eps = Q_ch_inv * Σ_L * randn(J-1)
        v_rnd = v + eps

        if maximum(v_rnd) < 0
            choice_rnd = base
        else
            choice_rnd = nonbase[argmax(v_rnd)]
        end
        probs[choice_rnd] += 1
    end
    probs ./= n_draws
    return probs
end

function predict_probs(
    data::NonHierData,
    β::Vector{Float64}, Σ::Symmetric{Float64,Array{Float64,2}},
    ν::Vector{Float64},
    n_draws::Int64
    )
    #Probabilities
    Σ_L = cholesky(Σ).L
    v = update_v(data, β)
    probs = zeros(data.J, data.N)
    for n in 1:data.N
        probs[:,n] = genrobit_probs_fs(
            v[:,n], Σ_L, ν, data.J, data.base, data.nonbase, n_draws
            )
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
    ν::Vector{Float64},
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
    probs1 = predict_probs(data1, β, Σ, ν, n_draws)[1]
    w1 = mean(probs1, dims=2)[:]

    w_δ = (w1 .- w0) ./ w0
    η = w_δ ./ δ
    #w_δ = (w1 .- w0) ./ (w1 .+ w0)
    #x_δ = δ / (2 + δ)
    #η = w_δ ./ x_δ

    Δ = mean(probs1 .- probs0, dims=2)[:]
    return η, Δ, probs1
end
