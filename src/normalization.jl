struct LinearNormalization{T} <: Any where {T}
    a::T
    b::T

    function LinearNormalization(a::T, b::T) where {T}
        return new{T}(deepcopy(a), deepcopy(b))
    end
end


function ZeroOneNormalization(ranges)
    range = ranges[:, 2] - ranges[:, 1]
    a = ranges[:, 1]
    b = convert(eltype(range), 1.0) ./ range
    return LinearNormalization(a, b)
end

function PosNegNormalization(ranges)
    range = ranges[:, 2] - ranges[:, 1]
    T = eltype(range)
    a = ranges[:, 1] .+ convert(T, 0.5) .* range
    b = convert(T, 2.0) ./ range
    return LinearNormalization(a, b)
end

# 2 (x - l) / r - 1
# 2 (x - l) / r - r/r
# (2x - 2l - r) / r
# 2(x - l - 0.5r) / r

function GaussianNormalization(μ, σ)
    a = μ
    b = convert(eltype(a), 1.0) ./ σ
    println("a ",a )
    println("b ", b)
    return LinearNormalization(a, b)
end

function (f::LinearNormalization)(x)
    y = zero(x)
    @. y = (x - f.a) * f.b
    return y
end

function (f::LinearNormalization)(buff::T, x::T) where {T}
    @. buff = (x - f.a) * f.b
    return buff
end

