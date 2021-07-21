"""
    gaussian_stats([::Type{T},] num_features::Int[, weight])

This function creates an OnlineStats.KahanVariance object for tracking the mean and variance for a vector.
Any OnlineStats.Weight can be used. The default is OnlineStats.EqualWeight and OnlineStats.ExponentialWeight 
if an integer or float is given as the weight. 
    
See also: [`extrema_stats`](@ref), [`LinearNormalization`](@ref)

# Examples
```jldoctest
julia> stats = gaussian_stats(Float32, 2, 1e-4)
Group
├─ KahanVariance: n=0 | value=1.0
└─ KahanVariance: n=0 | value=1.0

julia> fit!(stats, [1.0, 2.0])
Group
├─ KahanVariance: n=1 | value=1.0
└─ KahanVariance: n=1 | value=1.0

```
"""
function gaussian_stats end

function gaussian_stats(::Type{T}, num_features::Int, weight) where {T}
    return num_features*KahanVariance(T=T, weight=weight)
end

function gaussian_stats(::Type{T}, num_features::Int) where {T}
    return num_features*KahanVariance(T)
end

function gaussian_stats(::Type{T}, num_features::Int, weight::TW) where {T, TW<:Real}
    return gaussian_stats(T, num_features, ExponentialWeight(weight))
end

function gaussian_stats(num_features::Int, weight)
    return gaussian_stats(Float64, num_features, weight)
end

function gaussian_stats(num_features::Int)
    return gaussian_stats(Float64, num_features)
end

"""
    extrema_stats([::Type{T},] num_features::Int)

This function creates an OnlineStats.KahanVariance object for tracking the mean and variance for a vector.
Any OnlineStats.Weight can be used. The default is OnlineStats.EqualWeight and OnlineStats.ExponentialWeight 
if an integer or float is given as the weight. 
    
See also: [`extrema_stats`](@ref), [`LinearNormalization`](@ref)

# Examples
```jldoctest
julia> stats = gaussian_stats(Float32, 2, 1e-4)
Group
├─ KahanVariance: n=0 | value=1.0
└─ KahanVariance: n=0 | value=1.0

julia> fit!(stats, [1.0, 2.0])
Group
├─ KahanVariance: n=1 | value=1.0
└─ KahanVariance: n=1 | value=1.0

```
"""
function extrema_stats(::Type{T}, num_features::Int) where {T}
    return num_features*Extrema(T)
end

function extrema_stats(num_feature::Int)
    return num_feature*Extrema(Float64)
end

