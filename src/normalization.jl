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

struct GaussianNormalization{T} <: Any where {T}
    v::T

    function GaussianNormalization(num_features::Int)
        v = num_features*KahanVariance()
        return new{typeof(v)}(v)
    end

    function GaussianNormalization(num_features::Int, weight::T) where {T<:Real}
        v = num_features*KahanVariance(weight=Exponential(weight))
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
