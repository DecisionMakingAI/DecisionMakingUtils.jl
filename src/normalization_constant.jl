
"""
    LinearNormalization{T}(a::T,b::T)

This is a functor that normalizes a vector x as ``(x - a) * b``. This is the standard interface for all 
linear normalizations such as mapping to ``[0,1]``, ``[-1,1]`` and mean zero standard deviation one. 
LinearNormalization also supports the functions Base.lenght and Base.eltype.

See also: [`PosNegNormalization`](@ref), [`GaussianNormalization`](@ref)

# Examples
```jldoctest
julia> nrm = LinearNormalization([0.1, 2.0], [1.0, 0.5]);

julia> x = [1.0, 2.0];

julia> nrm(x)
2-element Vector{Float64}:
 0.9
 0.0

julia> nrm = LinearNormalization(2);  # no scaling to input vector

julia> nrm(x)
2-element Vector{Float64}:
 1.0
 2.0

julia> low = [0.0, -1.0];

julia> high = [3.0, 0.5];

julia> nrm = ZeroOneNormalization(low, high);  # normalize each entry to [0,1]

julia> x = [1.0, 0.0];

julia> feats = nrm(x)
2-element Array{Float64,1}:
0.3333333333333333
0.6666666666666666

julia> y = zero(x);  # create buffer to prevent allocations

julia> feats = nrm(y, x);  # no allocation return

julia> nrm = PosNegNormalization(low, high);  # normalize each entry to [0,1]

julia> nrm(x)
2-element Vector{Float64}:
 -0.3333333333333333
  0.3333333333333333

julia> μ = [0.0, 1.0];  # vector of means

julia> σ = [1.0, 2.0];  # vector of standard deviations

julia> nrm = GaussianNormalization(μ, σ);  # normalize x to be mean 0 and standard deviation 1

julia> nrm(x)
2-element Vector{Float64}:
  1.0
 -0.5
```
"""
struct LinearNormalization{T} <: Any where {T}
    a::T
    b::T

    function LinearNormalization(a::T, b::T) where {T}
        return new{T}(deepcopy(a), deepcopy(b))
    end

    function LinearNormalization(::Type{T}, num_features::Int) where {T}
        a = zeros(T, num_features)
        b = ones(T, num_features)
        return new{typeof(a)}(a,b)
    end

    function LinearNormalization(num_features::Int) where {T}
        LinearNormalization(Float64, num_features)
    end
end

function ZeroOneNormalization(low, high)
    a, b = zero(low), zero(low)
    @. b = one(eltype(low)) / high - low
    @. a = low
    return LinearNormalization(a,b)
end

function ZeroOneNormalization(ranges)
    return ZeroOneNormalization(ranges[:, 1], ranges[:, 2])
end

function PosNegNormalization(low, high)
    a, b = zero(low), zero(low)
    @. a = posneg_norm_a(eltype(low), low, high)
    @. b = posneg_norm_b(eltype(low), low, high)
    return LinearNormalization(a, b)
end

function PosNegNormalization(ranges)
    return PosNegNormalization(ranges[:, 1], ranges[:, 2])
end

function GaussianNormalization(μ, σ)
    a = μ
    b = one(eltype(a)) ./ σ
    return LinearNormalization(a, b)
end

function length(f::LinearNormalization)
    return length(f.a)
end

function eltype(f::LinearNormalization)
    return eltype(f.a)
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

function update!(f::LinearNormalization, a, b)
    @. f.a = a
    @. f.b = b
    return nothing
end

function update!(f::LinearNormalization, st::Group, mode=:default) where {TS<:Union{KahanVariance, Variance}}
    T = eltype(f)
    for i in 1:length(f)
        f.a[i], f.b[i] = update_linearnorm(T, st[i], mode)
    end
end

function update_linearnorm(::Type{T}, st::TS, mode=:default) where {T, TS<:Union{KahanVariance, Variance}}
    a = convert(T, mean(st))
    b = zero(T)
    if mode == :default
        b = one(T) / convert(T, std(st))
    elseif mode == :var
        b = one(T) / convert(T, var(st))
    else
        throw(DomainError(mode, "Not a valid mode for normalization with $TS"))
    end
    return a, b
end

function update_linearnorm(::Type{T}, st::TS, mode=:default) where {T,TS<:Extrema}
    a,b = zero(T), zero(T)
    if mode == :default
        a = convert(T, minimum(st))
        b = convert(T, maximum(st) - minimum(st))
    elseif mode == :posneg
        a = posneg_norm_a(T, minimum(st), maximum(st))
        b = posneg_norm_b(T, minimum(st), maximum(st))
    else
        throw(DomainError(mode, "Not a valid mode for normalization"))
    end
    return a,b
end

function posneg_norm_a(::Type{T}, low, high) where {T}
    return convert(T, 0.5 * (low + high)) 
end

function posneg_norm_b(::Type{T}, low, high) where {T}
    return convert(T, 2 / (high - low))
end