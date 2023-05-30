"""
    TileCodingBasis(num_inputs::Int, num_tiles::Int)

Creates a tile coding basis with `num_inputs` inputs and `num_tiles` tiles per input. 
Alternatively, construction can just specify `num_tiles` as a vector of integers, specifying the number of tiles per input.
The tiles are spread equally across [0,1] for each input. The basis is implemented as a sparse vector, where each element is 1 if the input is in the corresponding tile, and 0 otherwise.

See also: [`FourierBasisBuffer`](@ref)


# Examples
```jldoctest
julia> f = TileCodingBasis(2, 3);

julia> x = [0.0, 0.0];

julia> feats = f(x)
1

julia> feats = f([1,1])
9

julia> length(f)
9

julia> f = TileCodingBasis([10,10,3]);

julia> length(f)
300

```
"""
struct TileCodingBasis{T,TL,TC} <: Any where {T,TL,TC}
    bins::T
    num_inputs::Int
    lidxs::TL
    cidxs::TC
    function TileCodingBasis(num_inputs::Int, num_tiles::Int)
        bins = range(0.0, stop=1.0, length=num_tiles)
        sz = ntuple(i->length(bins), num_inputs)
        lidxs = LinearIndices(sz)
        cidxs = CartesianIndices(sz)
        new{typeof(bins), typeof(lidxs), typeof(cidxs)}(bins, num_inputs, lidxs, cidxs)
    end

    function TileCodingBasis(num_tiles::Vector{Int})
        bins = [range(0.0, stop=1.0, length=n) for n in num_tiles]
        sz = ntuple(i->length(bins[i]), length(num_tiles))
        lidxs = LinearIndices(sz)
        cidxs = CartesianIndices(sz)
        new{typeof(bins), typeof(lidxs), typeof(cidxs)}(bins, length(num_tiles), lidxs, cidxs)
    end
end


function (ϕ::TileCodingBasis{<:StepRangeLen})(x)
    n = length(ϕ.bins)
    f(i,x) = min(searchsortedfirst(ϕ.bins, x[i]), n)
    return ϕ.lidxs[CartesianIndex(ntuple(i->f(i,x), ϕ.num_inputs))]
end

function (ϕ::TileCodingBasis{<:Vector{<:StepRangeLen}})(x)
    f(i,x) = min(searchsortedfirst(ϕ.bins[i], x[i]), length(ϕ.bins[i]))
    return ϕ.lidxs[CartesianIndex(ntuple(i->f(i,x), ϕ.num_inputs))]
end

"""
    length(ϕ::TileCodingBasis)

Returns the number of feautes produced by the tile coding basis. 
"""

function Base.length(ϕ::TileCodingBasis{<:StepRangeLen})
    n = length(ϕ.bins)
    return n^ϕ.num_inputs
end

function Base.length(ϕ::TileCodingBasis{<:Vector{<:StepRangeLen}})
    return prod(length, ϕ.bins)
end

