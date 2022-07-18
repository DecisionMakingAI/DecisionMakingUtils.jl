"""
    TileCodingBasis([::Type,] num_inputs::Int, num_tiles::Int)

Creates a struct to generate fourier features up to a given order. Both coupled fourier features, e.g., 
``\\cos(π(3x₁ + 2x₂))`` and uncoupled features, ``[\\cos(πx₁), \\cos(π2x₁), …]`` can be generated with this basis function. 
The dorder parameter controls order for the coupling features. The number of coupled features generated, ``(\\text{dorder}+1)^{\\text{num\\_inputs}}``,
 grows exponentially with dorder, so it is not reccomended for use with high deminsional vectors.
The iorder parameter controls the order of the independent features generated. The full parameter determines if both 
``\\sin`` and ``\\cos`` features are generated, if false only cos features are generated. 

See also: [`FourierBasisBuffer`](@ref)


# Examples
```jldoctest
julia> f = TileCodingBasis(2, 3);

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
struct TileCodingBasis{T,TL,TC} <: Any where {T,TL,TC}
    bins::T
    num_inputs::Int
    lidxs::TL
    cidxs::TC
    function TileCodingBasis(num_inputs::Int, num_tiles::Int)
        bins = 0:1/num_tiles:1
        sz = ntuple(i->length(bins), num_inputs)
        lidxs = LinearIndices(sz)
        cidxs = CartesianIndices(sz)
        new{typeof(bins), typeof(lidxs), typeof(cidxs)}(bins, num_inputs, lidxs, cidxs)
    end
end


function (ϕ::TileCodingBasis{T,<:LinearIndices{N}})(x) where {T,N}
    n = length(ϕ.bins)
    f(i,x) = min(searchsortedfirst(ϕ.bins, x[i]), n)
    return ϕ.lidxs[CartesianIndex(ntuple(i->f(i,x), Val(N)))]
end

"""
    length(ϕ::TileCodingBasis)

Returns the number of feautes produced by the tile coding basis. 
"""

function length(ϕ::TileCodingBasis)
    n = length(ϕ.bins)
    return n^ϕ.num_inputs
end