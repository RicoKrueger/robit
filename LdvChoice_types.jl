abstract type Model end

struct Probit <: Model
    name::String

    function Probit(name::String)
        new("probit_$(name)")
    end
end

struct Robit <: Model
    name::String

    function Robit(name::String)
        new("robit_$(name)")
    end
end

struct GenRobit <: Model
    name::String

    function GenRobit(name::String)
        new("genrobit_$(name)")
    end
end

struct HierProbit <: Model
    name::String

    function HierProbit(name::String)
        new("hier_probit_$(name)")
    end
end

struct HierRobit <: Model
    name::String

    function HierRobit(name::String)
        new("hier_robit_$(name)")
    end
end

struct HierGenRobit <: Model
    name::String

    function HierGenRobit(name::String)
        new("hier_genrobit_$(name)")
    end
end

abstract type Data end

struct NonHierData <: Data
    N::Int64
    J::Int64
    base::Int64
    nonbase::Vector{Int64}
    choice::Vector{Int64}
    x::Array{Float64,3}
    label_x::Vector{String}
    K::Int64

    function NonHierData(
        df::DataFrame,
        label_alt_id::String,
        label_chosen::String,
        label_x::Vector{String},
        base::Int64
        )

        alt_id = Array(df[!,label_alt_id])
        chosen = BitArray(df[!,label_chosen])
        choice = alt_id[chosen]
        x = Array(df[!,label_x])

        J = maximum(alt_id)
        N = Int(size(df)[1] / J)
        K = length(label_x)
        nonbase = [j for j in 1:J if j != base]

        mask_base = alt_id .== base
        mask_nonbase = BitArray(map((x) -> x ∈ nonbase, alt_id))

        x_base = repeat(x[mask_base,:], inner=(J-1,1))
        x = x[mask_nonbase,:] .- x_base
        x = reshape(x', (K, J-1, N))

        new(N, J, base, nonbase, choice, x, label_x, K)
    end
end

function map_ind_obs(
    ind_id::Vector{Int64}, obs_id::Vector{Int64}, mask_base::BitArray{1}
    )
    n_ind = length(unique(ind_id))
    N = length(unique(obs_id))

    ind_id_obs = copy(ind_id[mask_base])
    obs_id_obs = copy(obs_id[mask_base])
    obs_to_ind = ones(N)
    c = 1
    for n in 2:N
        if ind_id_obs[n] != ind_id_obs[n - 1]
            c += 1
        end
        obs_to_ind[n] = c
    end

    obs_per_ind = zeros(n_ind)
    i = 1
    for n in 1:N
        if n > 1
            if ind_id_obs[n] != ind_id_obs[n - 1]
                i += 1
            end
        end
        obs_per_ind[i] += 1
    end

    return n_ind, obs_to_ind, obs_per_ind
end

struct HierData <: Data
    N::Int64
    J::Int64
    n_ind::Int64
    obs_to_ind::Vector{Int64}
    obs_per_ind::Vector{Int64}
    base::Int64
    nonbase::Vector{Int64}
    choice::Vector{Int64}
    x_fix::Array{Float64,3}
    x_rnd::Array{Float64,3}
    label_x_fix::Vector{String}
    label_x_rnd::Vector{String}
    K_fix::Int64
    K_rnd::Int64

    function HierData(
        df::DataFrame,
        label_ind_id::String,
        label_obs_id::String,
        label_alt_id::String,
        label_chosen::String,
        label_x_fix::Vector{String},
        label_x_rnd::Vector{String},
        base::Int64
        )

        ind_id = Array(df[!,label_ind_id])
        obs_id = Array(df[!,label_obs_id])
        alt_id = Array(df[!,label_alt_id])
        chosen = BitArray(df[!,label_chosen])
        choice = copy(alt_id[chosen])
        x_fix = Array(df[!,label_x_fix])
        x_rnd = Array(df[!,label_x_rnd])

        J = maximum(alt_id)
        N = Int(size(df)[1] / J)
        K_fix = length(label_x_fix)
        K_rnd = length(label_x_rnd)
        nonbase = [j for j in 1:J if j != base]

        mask_base = alt_id .== base
        mask_nonbase = BitArray(map((x) -> x ∈ nonbase, alt_id))

        n_ind, obs_to_ind, obs_per_ind = map_ind_obs(ind_id, obs_id, mask_base)

        x_fix_base = repeat(x_fix[mask_base,:], inner=(J-1,1))
        x_fix = x_fix[mask_nonbase,:] .- x_fix_base
        x_fix = reshape(x_fix', (K_fix, J-1, N))

        x_rnd_base = repeat(x_rnd[mask_base,:], inner=(J-1,1))
        x_rnd = x_rnd[mask_nonbase,:] .- x_rnd_base
        x_rnd = reshape(x_rnd', (K_rnd, J-1, N))

        new(N, J, n_ind, obs_to_ind, obs_per_ind,
            base, nonbase, choice,
            x_fix, x_rnd,
            label_x_fix, label_x_rnd,
            K_fix, K_rnd)
    end
end

struct McmcArgs
    n_chain::Int64
    n_burn::Int64
    n_sample::Int64
    n_iter::Int64
    n_thin::Int64
    n_keep::Int64
    disp::Int64

    function McmcArgs(
        ; n_chain=1, n_burn=100, n_sample=100,n_thin=1,
        disp=100
        )

        n_iter = n_burn + n_sample
        n_keep = Int(n_sample / n_thin)

        new(n_chain, n_burn, n_sample, n_iter, n_thin, n_keep, disp)
    end
end
