using Random, Distributions, LinearAlgebra, Statistics, StatsBase
using DataFrames, CSV
using Revise

Random.seed!(1234)

function generate_data_probit(N, J)
    #Scenario
    #J alternatives
    #4 attributes

    #IDs
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    β = [1.0, -1.0, 1.0, -1.0]

    if J == 3
        α = [0.5, 0.5]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.8, 0.7])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    elseif J == 4
        α = [0.5, 0.5, -0.5]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.2, 1.0, 0.8])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    end

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x = 2 .* rand(N * J, length(β))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_diff = x[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_diff -= repeat(x[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v = reshape((x_diff * β + asc_diff * α), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    eps = cholesky(Σ).L * randn(J-1, N)
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to probit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    df = DataFrame(
        obs_id = obs_id,
        alt_id = alt_id,
        chosen = reshape(chosen, (J * N)),
        x1 = reshape(x[:,1], (J * N)),
        x2 = reshape(x[:,2], (J * N)),
        x3 = reshape(x[:,3], (J * N)),
        x4 = reshape(x[:,4], (J * N))
        )

    return df
end

function generate_data_robit(N, J)
    #Scenario
    #J alternatives
    #4 attributes

    #IDs
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    if J == 3
        α = [1.0, -1.0]
        β = [1.0, -1.0]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.0, 1.0])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    elseif J == 4
        α = [-1.0, 1.0, -1.0]
        β = [1.0, -1.0] #, 1.0, -1.0]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.0, 1.0, 1.0])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    end
    ν = 2.0

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x = 5 .* rand(N * J, length(β))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_diff = x[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_diff -= repeat(x[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v = reshape((x_diff * β + asc_diff * α), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    q = rand(Chisq(ν), N) ./ ν
    eps = zeros(J-1, N)
    for n in 1:N
        eps[:,n] = cholesky(Σ ./ q[n]).L * randn(J-1)
    end
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to robit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    if length(β) == 2
        df = DataFrame(
            obs_id = obs_id,
            alt_id = alt_id,
            chosen = reshape(chosen, (J * N)),
            x1 = reshape(x[:,1], (J * N)),
            x2 = reshape(x[:,2], (J * N))
            )
    elseif length(β) == 4
        df = DataFrame(
            obs_id = obs_id,
            alt_id = alt_id,
            chosen = reshape(chosen, (J * N)),
            x1 = reshape(x[:,1], (J * N)),
            x2 = reshape(x[:,2], (J * N)),
            x3 = reshape(x[:,3], (J * N)),
            x4 = reshape(x[:,4], (J * N))
            )
    end

    return df, q
end

function generate_data_genrobit(N, J)
    #Scenario
    #J alternatives
    #4 attributes

    #IDs
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    if J == 3
        α = [1.0, -1.0]
        β = [1.0, -1.0]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.0, 1.0])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
        ν = [3.0, 1.0]
    elseif J == 4
        α = [-1.0, 1.0, -1.0]
        β = [1.0, -1.0] #, 1.0, -1.0]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.0, 1.0, 1.0])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
        ν = [5.0, 3.0, 1.0]
    end

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x = 6 .* rand(N * J, length(β))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_diff = x[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_diff -= repeat(x[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v = reshape((x_diff * β + asc_diff * α), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    q = zeros(J-1,N)
    for j in 1:(J-1)
        q[j,:] = rand(Chisq(ν[j]), N) ./ ν[j]
    end

    eps = zeros(J-1, N)
    for n in 1:N
        Q_ch_inv = Diagonal(1 ./ sqrt.(q[:,n]))
        eps[:,n] = Q_ch_inv * cholesky(Σ).L * randn(J-1)
    end
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to Gen-robit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    if length(β) == 2
        df = DataFrame(
            obs_id = obs_id,
            alt_id = alt_id,
            chosen = reshape(chosen, (J * N)),
            x1 = reshape(x[:,1], (J * N)),
            x2 = reshape(x[:,2], (J * N))
            )
    elseif length(β) == 4
        df = DataFrame(
            obs_id = obs_id,
            alt_id = alt_id,
            chosen = reshape(chosen, (J * N)),
            x1 = reshape(x[:,1], (J * N)),
            x2 = reshape(x[:,2], (J * N)),
            x3 = reshape(x[:,3], (J * N)),
            x4 = reshape(x[:,4], (J * N))
            )
    end

    return df, q
end

function generate_data_hier_probit(n_ind, T, J)
    #Scenario
    #n_ind individuals
    #T observations per individual
    #J alternatives
    #4 attributes

    obs_per_ind = 1 .+ rand(Poisson(T), n_ind)
    N = sum(obs_per_ind)

    #IDs
    ind_id_obs = zeros(Int64, N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            ind_id_obs[n] = i
        end
    end
    ind_id = repeat(ind_id_obs, inner=J)
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    β = [1.0, -1.0]
    η = [1.0, -1.0]
    ω = [1.0,  1.0]
    γ_i = repeat(η, 1, n_ind) .+ repeat(ω, 1, n_ind) .* randn(length(η), n_ind)
    γ = zeros(length(η), N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            γ[:,n] = γ_i[:,i]
        end
    end

    if J == 3
        α = [0.5, 0.5]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.8, 0.7])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    elseif J == 4
        α = [0.5, 0.5, -0.5]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.0, 1.0, 1.0])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    end

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x_fix = 2 .* rand(N * J, length(β))
    x_rnd = 2 .* rand(N * J, length(η))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_fix_diff = x_fix[alt_id .!= base,:]
    x_rnd_diff = x_rnd[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_fix_diff -= repeat(x_fix[alt_id .== base,:], inner=(J-1,1))
    x_rnd_diff -= repeat(x_rnd[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v_fix = x_fix_diff * β
    v_rnd = sum(x_rnd_diff .* repeat(γ', inner=(J-1,1)), dims=2)[:]
    v = reshape((asc_diff * α .+ v_fix .+ v_rnd), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    eps = cholesky(Σ).L * randn(J-1, N)
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to hierarchical probit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    df = DataFrame(
        ind_id = ind_id,
        obs_id = obs_id,
        alt_id = alt_id,
        chosen = reshape(chosen, (J * N)),
        x1 = reshape(x_fix[:,1], (J * N)),
        x2 = reshape(x_fix[:,2], (J * N)),
        x3 = reshape(x_rnd[:,1], (J * N)),
        x4 = reshape(x_rnd[:,2], (J * N))
        )

    return df
end

function generate_data_hier_robit(n_ind, T, J)
    #Scenario
    #n_ind individuals
    #T observations per individual
    #J alternatives
    #4 attributes

    obs_per_ind = 1 .+ rand(Poisson(T), n_ind)
    N = sum(obs_per_ind)

    #IDs
    ind_id_obs = zeros(Int64, N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            ind_id_obs[n] = i
        end
    end
    ind_id = repeat(ind_id_obs, inner=J)
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    β = [1.0, -1.0]
    η = [1.0, -1.0]
    ω = [1.0,  1.0]
    γ_i = repeat(η, 1, n_ind) .+ repeat(ω, 1, n_ind) .* randn(length(η), n_ind)
    γ = zeros(length(η), N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            γ[:,n] = γ_i[:,i]
        end
    end
    ν = 2.0

    if J == 3
        α = [0.5, 0.5]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.8, 0.7])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    elseif J == 4
        α = [0.5, 0.5, -0.5]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.2, 1.0, 0.8])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
    end

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x_fix = 2 .* rand(N * J, length(β))
    x_rnd = 2 .* rand(N * J, length(η))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_fix_diff = x_fix[alt_id .!= base,:]
    x_rnd_diff = x_rnd[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_fix_diff -= repeat(x_fix[alt_id .== base,:], inner=(J-1,1))
    x_rnd_diff -= repeat(x_rnd[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v_fix = x_fix_diff * β
    v_rnd = sum(x_rnd_diff .* repeat(γ', inner=(J-1,1)), dims=2)[:]
    v = reshape((asc_diff * α .+ v_fix .+ v_rnd), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    q = rand(Chisq(ν), N) ./ ν
    eps = zeros(J-1, N)
    for n in 1:N
        eps[:,n] = cholesky(Σ ./ q[n]).L * randn(J-1)
    end
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to hierarchical robit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    df = DataFrame(
        ind_id = ind_id,
        obs_id = obs_id,
        alt_id = alt_id,
        chosen = reshape(chosen, (J * N)),
        x1 = reshape(x_fix[:,1], (J * N)),
        x2 = reshape(x_fix[:,2], (J * N)),
        x3 = reshape(x_rnd[:,1], (J * N)),
        x4 = reshape(x_rnd[:,2], (J * N))
        )

    return df
end

function generate_data_hier_genrobit(n_ind, T, J)
    #Scenario
    #n_ind individuals
    #T observations per individual
    #J alternatives
    #4 attributes

    obs_per_ind = 1 .+ rand(Poisson(T), n_ind)
    N = sum(obs_per_ind)

    #IDs
    ind_id_obs = zeros(Int64, N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            ind_id_obs[n] = i
        end
    end
    ind_id = repeat(ind_id_obs, inner=J)
    obs_id = repeat(collect(1:N), inner=J)
    alt_id = repeat(collect(1:J), outer=N)

    #Base and nonbase alternatives
    base = J
    nonbase = [j for j in 1:J if j != base]

    #True parameter values
    β = [1.0, -1.0]
    η = [1.0, -1.0]
    ω = [1.0,  1.0]
    γ_i = repeat(η, 1, n_ind) .+ repeat(ω, 1, n_ind) .* randn(length(η), n_ind)
    γ = zeros(length(η), N)
    n = 0
    for i in 1:n_ind
        for t in 1:obs_per_ind[i]
            n += 1
            γ[:,n] = γ_i[:,i]
        end
    end
    ν = 2.0

    if J == 3
        α = [0.5, 0.5]
        Ω = Symmetric([1.0 0.3; 0.3 1.0])
        σ = sqrt.([1.8, 0.7])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
        ν = [3.0, 1.0]
    elseif J == 4
        α = [0.5, 0.5, -0.5]
        Ω = Symmetric([1.0 0.3 0.0;
                       0.3 1.0 0.3;
                       0.0 0.3 1.0])
        σ = sqrt.([1.2, 1.0, 0.8])
        D = Diagonal(σ)
        Σ = Symmetric(D * Ω * D)
        ν = [5.0, 3.0, 1.0]
    end

    #ASCs and attributes
    asc = zeros(N * J, J - 1)
    for (i, j) in enumerate(nonbase)
        asc[alt_id .== j,i] .= 1
    end

    x_fix = 2 .* rand(N * J, length(β))
    x_rnd = 2 .* rand(N * J, length(η))

    #Differenced attributes
    asc_diff = asc[alt_id .!= base,:]
    x_fix_diff = x_fix[alt_id .!= base,:]
    x_rnd_diff = x_rnd[alt_id .!= base,:]

    asc_diff -= repeat(asc[alt_id .== base,:], inner=(J-1,1))
    x_fix_diff -= repeat(x_fix[alt_id .== base,:], inner=(J-1,1))
    x_rnd_diff -= repeat(x_rnd[alt_id .== base,:], inner=(J-1,1))

    #Deterministic utility
    v_fix = x_fix_diff * β
    v_rnd = sum(x_rnd_diff .* repeat(γ', inner=(J-1,1)), dims=2)[:]
    v = reshape((asc_diff * α .+ v_fix .+ v_rnd), (J-1,N))

    v_det = zeros(J,N)
    v_det[nonbase,:] = v

    #Random utility
    q = zeros(J-1,N)
    for j in 1:(J-1)
        q[j,:] = rand(Chisq(ν[j]), N) ./ ν[j]
    end

    eps = zeros(J-1, N)
    for n in 1:N
        Q_ch_inv = Diagonal(1 ./ sqrt.(q[:,n]))
        eps[:,n] = Q_ch_inv * cholesky(Σ).L * randn(J-1)
    end
    v_eps = v + eps

    #Choices
    choice_rnd = zeros(Int64, N)
    choice_err = zeros(Int64, N)
    chosen = zeros(Int64, (J, N))

    for n in 1:N
        if maximum(v_eps[:,n]) < 0
            choice_rnd[n] = base
        else
            choice_rnd[n] = nonbase[argmax(v_eps[:,n])]
        end

        chosen[choice_rnd[n],n] = 1
        choice_det = findall(v_det[:,n] .==  maximum(v_det[:,n]))
        choice_err[n] = Int(choice_rnd[n] ∉ choice_det)
    end

    println(" ")
    println("Generating data according to hierarchical Gen-robit model.")
    println("Error rate: ", mean(choice_err))
    println("Market shares: ")
    println(proportionmap(choice_rnd))

    #Data frame
    df = DataFrame(
        ind_id = ind_id,
        obs_id = obs_id,
        alt_id = alt_id,
        chosen = reshape(chosen, (J * N)),
        x1 = reshape(x_fix[:,1], (J * N)),
        x2 = reshape(x_fix[:,2], (J * N)),
        x3 = reshape(x_rnd[:,1], (J * N)),
        x4 = reshape(x_rnd[:,2], (J * N))
        )

    return df
end

N = 10000
J = 4


#df = generate_data_probit(N, J)
#CSV.write("synthetic_data_probit.csv", df)

df, q = generate_data_robit(N, J)
CSV.write("synthetic_data_robit.csv", df)

df, q = generate_data_genrobit(N, J)
CSV.write("synthetic_data_genrobit.csv", df)

#=
n_ind = 500
T = 5
J= 4

df = generate_data_hier_probit(n_ind, T, J)
CSV.write("synthetic_data_hier_probit_panel.csv", df)

df = generate_data_hier_robit(n_ind, T, J)
CSV.write("synthetic_data_hier_robit_panel.csv", df)

df = generate_data_hier_genrobit(n_ind, T, J)
CSV.write("synthetic_data_hier_genrobit_panel.csv", df)
=#
