"""
    FourierBasis([::Type,] num_inputs::Int, dorder::Int, iorder::Int [, full::Bool=false])

Creates a struct to generate fourier features up to a given order. Both coupled fourier features, e.g., 
``\\cos(π(3x₁ + 2x₂))`` and uncoupled features, ``[\\cos(πx₁), \\cos(π2x₁), …]`` can be generated with this basis function. 
The dorder parameter controls order for the coupling features. The number of coupled features generated, ``(\\text{num\\_inputs}+1)^{\\text{dorder}}``,
 grows exponentially with dorder, so it is not reccomended for use with high deminsional vectors.
The iorder parameter controls the order of the independent features generated. The full parameter determines if both 
``\\sin`` and ``\\cos`` features are generated, if false only cos features are generated. 

See also: [`FourierBasisBuffer`](@ref)


# Examples
```jldoctest
julia> f = FourierBasis(2, 1, 2);

julia> x = [0.0, 0.5];

julia> feats = f(x)
6-element Array{Float64,1}:
  1.0
  6.123233995736766e-17
  1.0
  6.123233995736766e-17
  1.0
 -1.0

```
"""
struct FourierBasis{T,TF} <: Any where {T,TF}
    C::T
    
    function FourierBasis(::Type{T}, num_inputs::Int, dorder::Int, iorder::Int, both::Bool=false) where {T}
        C = make_Cmat(T, num_inputs, dorder, iorder)
        new{typeof(C),both}(C)
    end
    
    function FourierBasis(num_inputs::Int, dorder::Int, iorder::Int, both::Bool=false) where {T}
        return FourierBasis(Float64, num_inputs, dorder, iorder, both)
    end
end

"""
    FourierBasisBuffer(ϕ::FourierBasis)

Creates preallocated buffers for a fourier basis to use to avoid making allocations on every call of the basis. 
    
See also: [`FourierBasis`](@ref)

# Examples
```jldoctest
julia> f = FourierBasis(2, 1, 2);

julia> buff = FourierBasisBuffer(f);

julia> x = [0.0, 0.5];

julia> feats = f(buff, x)
6-element Array{Float64,1}:
  1.0
  6.123233995736766e-17
  1.0
  6.123233995736766e-17
  1.0
 -1.0
"""
struct FourierBasisBuffer{T} <: Any 
    y::T
    z::T

    function FourierBasisBuffer(ϕ::FourierBasis{TC,TB}) where {TC, TB}
        T = eltype(ϕ.C)
        n = size(ϕ.C, 2)
        y = zeros(T, n)
        if TB == true
            z = zeros(T, n*2)
        else
            z = zeros(T, n)
        end
        return new{typeof(y)}(y, z)
    end
end

function increment_counter!(counter::Vector{Int}, maxDigit::Int)
    for i in length(counter):-1:1
        counter[i] += 1
        if (counter[i] > maxDigit)
            counter[i] = 0
        else
            break
        end
    end
end

function make_Cmat(T::Type, num_inputs::Int, dorder::Int, iorder::Int)
    iTerms = iorder * num_inputs
    dTerms = (dorder+1)^num_inputs
    oTerms = min(iorder, dorder) * num_inputs
    num_feats = iTerms + dTerms - oTerms
    C = zeros(T, (num_inputs, num_feats))
    counter = zeros(Int, num_inputs)
    termCount::Int = 1

    while termCount <= dTerms
        for i in 1:num_inputs
            C[i, termCount] = counter[i]
        end
        increment_counter!(counter, dorder)
        termCount += 1
    end
    for i in 1:num_inputs
        for j in (dorder+1):(iorder)
            C[i, termCount] = j
            termCount += 1
        end
    end
    C .*= π

    return C
end


function (ϕ::FourierBasis{T,false})(x) where {T}
    y = zeros(size(ϕ.C, 2))
    mul!(y, ϕ.C', x)
    @. y = cos(y)
    return y
end

function (ϕ::FourierBasis{T,false})(buff::FourierBasisBuffer, x) where {T}
    y = buff.y
    mul!(y, ϕ.C', x)
    @. y = cos(y)
    return y
end

function (ϕ::FourierBasis{T,true})(x) where {T}
    n = size(ϕ.C, 2)
    y = zeros(n)
    z = zeros(n*2)
    mul!(y, ϕ.C', x)
    @. z[1:n] = cos(y)
    @. z[n+1:end] = sin(y)
    return z
end

function (ϕ::FourierBasis{T,true})(buff::FourierBasisBuffer, x) where {T}
    y, z = buff.y, buff.z
    n = length(y)
    mul!(y, ϕ.C', x)
    @. z[1:n] = cos(y)
    @. z[n+1:end] = sin(y)
    return z
end


"""
    length(ϕ::FourierBasis)

Returns the number of feautes produced by the Fourier basis. 
"""
function length(ϕ::FourierBasis{T,false}) where {T}
    return size(ϕ.C, 2)
end

function length(ϕ::FourierBasis{T,true}) where {T}
    return size(ϕ.C, 2)*2
end

