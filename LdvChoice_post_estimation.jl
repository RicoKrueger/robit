function rhat(draws::Array{Float64,2})
    D0, C0 = size(draws)
    D = Int(D0 / 2)
    C = Int(C0 * 2)
    ψ_dc = reshape(draws, (D,C))

    ψ_c = mean(ψ_dc, dims=1)
    ψ = mean(ψ_c)
    B = D / (C - 1) * sum((ψ_c .- ψ).^2)
    s2_c = 1 ./ (D - 1) .* sum((ψ_dc .- ψ_c).^2, dims=1)[:]
    W = mean(s2_c)
    var = (D - 1) / D * W + 1 / D * B
    R = sqrt(var / W)
    return R
end

function post_summary(
    draws::Array{Float64,3}, name::String, labels::Vector{String}
    )
    K = size(draws)[1]

    df = DataFrame()
    df.name = labels
    df.mean = mean(draws, dims=(2,3))[:]
    df.std_dev = std(draws, dims=(2,3))[:]
    df.ci025 = [quantile(vec(draws[k,:,:]), 0.025) for k in 1:K]
    df.ci975 = [quantile(vec(draws[k,:,:]), 0.975) for k in 1:K]
    df.rhat = [rhat(draws[k,:,:]) for k in 1:K]

    println(" ")
    println(name, ":")
    println(df)
    return df
end

function post_summary(draws::Array{Float64,4}, nonbase::Vector{Int64})
    K = size(draws)[1]

    df = DataFrame()
    df.name = ["$(i) vs. $(j)" for i in nonbase for j in nonbase]
    df.mean = [mean(draws[i,j,:,:]) for i in 1:K for j in 1:K]
    df.std_dev = [std(draws[i,j,:,:]) for i in 1:K for j in 1:K]
    df.ci025 = [quantile(vec(draws[i,j,:,:]), 0.025) for i in 1:K for j in 1:K]
    df.ci975 = [quantile(vec(draws[i,j,:,:]), 0.975) for i in 1:K for j in 1:K]
    df.rhat = [rhat(draws[i,j,:,:]) for i in 1:K for j in 1:K]

    println(" ")
    println("Sigma:")
    println(df)
    return df
end

function post_summary(draws::Array{Float64,2})
    df = DataFrame()
    df.mean = [mean(draws)]
    df.std_dev = [std(draws)]
    df.ci025 = [quantile(vec(draws), 0.025)]
    df.ci975 = [quantile(vec(draws), 0.975)]
    df.rhat = [rhat(draws)]

    println(" ")
    println("nu:")
    println(df)
    return df
end
