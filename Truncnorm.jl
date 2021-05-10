module Truncnorm
export truncnorm_above_rnd, truncnorm_below_rnd

using SpecialFunctions

function normcdf(x::Float64)
    return 0.5 * erfc(-x / sqrt(2))
end

function invnormcdf(p::Float64)
    return -sqrt(2) * erfcinv(2 * p)
end

function truncnorm_below_rnd_tail(a::Float64)
    c = 0.5 .* a.^2
    while true
        u = rand()
        v = rand()
        w = c - log(u)
        if (v.^2 .* w) <= a
            return sqrt(2 .* w)
        end
    end
end

function truncnorm_below_rnd(μ::Float64, σ::Float64, a::Float64)
    if a - μ > 5 .* σ
        x = μ + σ * truncnorm_below_rnd_tail((a - μ) / σ)
    else
        c = normcdf((a - μ) / σ)
        u = c + (1 - c) .* rand()
        x = μ + σ .* invnormcdf(u)
    end
    return x
end

function truncnorm_above_rnd(μ::Float64, σ::Float64, b::Float64)
    if μ - b > 5 .* σ
        x = μ - σ .* truncnorm_below_rnd_tail((μ - b) / σ)
    else
        c = normcdf((b - μ) / σ)
        u = c .* rand()
        x = μ + σ .* invnormcdf(u)
    end
    return x
end

end
