"""
    ZeroOneNormalization(low::T, high::T)

This is a functor that normalizes a vector to be in the range [0,1]. Initial upper and lower bounds for each element are needed. 
    
See also: [`PosNegNormalization`](@ref), [`GaussianNormalization`](@ref)

# Examples
```jldoctest
julia> low = [0.0, -1.0];

julia> high = [3.0, 0.5];

julia> nrm = ZeroOneNormalization(low, high);

julia> x = [1.0, 0.0];

julia> feats = nrm(x)
2-element Array{Float64,1}:
 0.3333333333333333
 0.6666666666666666

julia> y = zero(x);  # create buffer to prevent allocations

julia> feats = nrm(y, x);  # no allocation return

julia> feats = nrm(y, x, fit=true);  # update upper and lower bounds if x is outside the range

```
"""
struct ZeroOneNormalization{T} <: Any where {T}
    low::T
    high::T 

    function ZeroOneNormalization(ranges)
        low = ranges[:, 1]
        high = ranges[:, 2]
        return new{typeof(low)}(low, high)
    end

    function ZeroOneNormalization(low::T, high::T) where {T}
        return new{T}(deepcopy(low), deepcopy(high))
    end
end

"""
    PosNegNormalization(low::T, high::T)

This is a functor that normalizes a vector to be in the range [-1,1]. Initial upper and lower bounds for each element are needed. 
    
See also: [`ZeroOneNormalization`](@ref), [`GaussianNormalization`](@ref)

# Examples
```jldoctest
julia> low = [0.0, -1.0];

julia> high = [3.0, 0.5];

julia> nrm = PosNegNormalization(low, high);

julia> x = [1.0, 0.0];

julia> feats = nrm(x)
2-element Array{Float64,1}:
 -0.33333333333333337
  0.33333333333333326

julia> y = zero(x);  # create buffer to prevent allocations

julia> feats = nrm(y, x);  # no allocation return

julia> feats = nrm(y, x, fit=true);  # update upper and lower bounds if x is outside the range

```
"""
struct PosNegNormalization{T} <: Any where {T}
    low::T
    high::T 

    function PosNegNormalization(ranges)
        low = ranges[:, 1]
        high = ranges[:, 2]
        return new{typeof(low)}(low, high)
    end

    function PosNegNormalization(low::T, high::T) where {T}
        return new{T}(deepcopy(low), deepcopy(high))
    end
end

"""
    GaussianNormalization(num_features::Int[, weight])

This is a functor that normalizes each element of a vector to be centered around a mean and have a variance of one. 
OnlineStats.KahanVariance is used for computing and tracking the mean and variance. Likewise any OnlineStats.Weight can
be used to weigh each sample. 
    
See also: [`ZeroOneNormalization`](@ref), [`PosNegNormalization`](@ref)

# Examples
```jldoctest
julia> nrm = GaussianNormalization(2, 1e-4);

julia> x = [1.0, 0.0];

julia> feats = nrm(x)
2-element Array{Float64,1}:
 1.0
 0.0

julia> y = zero(x);  # create buffer to prevent allocations

julia> feats = nrm(y, x);  # no allocation return

julia> feats = nrm(y, x, fit=true)  # update mean and variance using x
2-element Array{Float64,1}:
 0.9999
 0.0

```
"""
struct GaussianNormalization{T} <: Any where {T}
    v::T

    function GaussianNormalization(num_features::Int)
        v = num_features*KahanVariance()
        return new{typeof(v)}(v)
    end

    function GaussianNormalization(num_features::Int, weight::T) where {T<:Real}
        v = num_features*KahanVariance(weight=ExponentialWeight(weight))
        return new{typeof(v)}(v)
    end

    function GaussianNormalization(num_features::Int, weight::OnlineStats.Weight)
        v = num_features*KahanVariance(weight=weight)
        return new{typeof(v)}(v)
    end
end

function (f::ZeroOneNormalization)(x; fit=false)
    if fit
        @. f.low = min(x, f.low)
        @. f.high = max(x, f.high)
    end
    y = zero(x)
    @. y = (x - f.low) / (f.high - f.low)
    return y
end

function (f::ZeroOneNormalization)(buff::T, x::T; fit=false) where {T}
    if fit
        @. f.low = min(x, f.low)
        @. f.high = max(x, f.high)
    end
    @. buff = (x - f.low) / (f.high - f.low)
    return buff
end

function (f::PosNegNormalization)(x; fit=false)
    if fit
        @. f.low = min(x, f.low)
        @. f.high = max(x, f.high)
    end
    y = zero(x)
    T2 = eltype(x)
    a = convert(T2, 2.0)
    b = convert(T2, 1.0)
    @. y = a * (x - f.low) / (f.high - f.low) - b
    return y
end

function (f::PosNegNormalization)(buff::T, x::T; fit=false) where {T}
    if fit
        @. f.low = min(x, f.low)
        @. f.high = max(x, f.high)
    end
    T2 = eltype(x)
    a = convert(T2, 2.0)
    b = convert(T2, 1.0)
    @. buff = a * (x - f.low) / (f.high - f.low) - b
    return buff
end

function (f::GaussianNormalization)(x; fit=false)
    if fit
        fit!(f.v, x)
    end
    y = zero(x)
    T2 = eltype(x)
    @. y = (x - convert(T2, mean(f.v.stats))) / convert(T2, std(f.v.stats))
    return y
end

function (f::GaussianNormalization)(buff::T, x::T; fit=false) where {T}
    if fit
        fit!(f.v, x)
    end
    T2 = eltype(x)
    @. buff = (x - convert(T2, mean(f.v.stats))) / convert(T2, std(f.v.stats))
    return buff
end
