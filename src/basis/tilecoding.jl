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
    num_tilings::Int
    num_inputs::Int
    tiling_type::Symbol
    lidxs::TL
    cidxs::TC
    function TileCodingBasis(num_inputs::Int, num_tiles::Int; num_tilings::Int, tiling_type::Symbol=:wrap, tile_loc=:equal)
        @assert tile_loc == :equal || tile_loc == :random
        if tile_loc == :equal
            bins = range(0.0, stop=1.0, length=num_tiles+1)[2:end]
        else
            bins = rand(num_tiles, num_inputs, num_tilings)
            @. bins[end,:,:] = 1
        end
        sz = ntuple(i->num_tiles, num_inputs)
        lidxs = LinearIndices(sz)
        cidxs = CartesianIndices(sz)
        @assert tiling_type == :wrap || tiling_type == :clip
        new{typeof(bins), typeof(lidxs), typeof(cidxs)}(bins, num_tilings, num_inputs, tiling_type, lidxs, cidxs)
    end

    function TileCodingBasis(tiles_per_dim::Vector{Int}; num_tilings::Int, tiling_type::Symbol=:wrap, tile_loc=:equal)
        @assert tile_loc == :equal || tile_loc == :random
        if tile_loc == :equal
            bins = [range(0.0, stop=1.0, length=n+1)[2:end] for n in tiles_per_dim]
        else
            bins = [[rand(n) for n in tiles_per_dim] for _ in 1:num_tilings]
            for i in 1:num_tilings
                for j in 1:length(bins[i])
                    bins[i][j][end] = 1
                end
            end
        end
        sz = ntuple(i->tiles_per_dim[i], length(tiles_per_dim))
        lidxs = LinearIndices(sz)
        cidxs = CartesianIndices(sz)
        @assert tiling_type == :wrap || tiling_type == :clip
        new{typeof(bins), typeof(lidxs), typeof(cidxs)}(bins, num_tilings, length(tiles_per_dim), tiling_type, lidxs, cidxs)
    end
end

function tile_index(bins, x, offset)
    n = length(bins)
    idx = min(searchsortedfirst(bins, x + offset), n)
    return idx
end

function tile_index_wrap(bins, x, offset)
    n = length(bins)
    idx = min(searchsortedfirst(bins, (x + offset) % 1.0), n)
    return idx
end

function get_tiling(bins::T, lidxs, num_inputs, x, offset, index_func) where {T <: StepRangeLen}
    return lidxs[CartesianIndex(ntuple(i->index_func(bins, x[i], offset), num_inputs))]
end

function get_tiling(bins::Vector{<:StepRangeLen}, lidxs, num_inputs, x, offset, index_func)
    return lidxs[CartesianIndex(ntuple(i->index_func(bins[i], x[i], offset[i]), num_inputs))]
end

function get_tiling(bins::AbstractArray{T,2}, lidxs, num_inputs, x, index_func) where {T}

    return lidxs[CartesianIndex(ntuple(i->index_func(view(bins, :, i), x[i], zero(eltype(x))), num_inputs))]
end

function get_tiling(bins::Vector{Vector{T}}, lidxs, num_inputs, x, index_func) where {T}
    return lidxs[CartesianIndex(ntuple(i->index_func(bins[i], x[i], zero(eltype(x))), num_inputs))]
end


function (ϕ::TileCodingBasis{<:StepRangeLen})(x)
    offsets = (0:ϕ.num_tilings-1) ./ ϕ.num_tilings .* (1 / length(ϕ.bins)) # no allocations :) 
    if ϕ.tiling_type == :wrap
        fwrap(i,x) = get_tiling(ϕ.bins, ϕ.lidxs, ϕ.num_inputs, x, offsets[i], tile_index_wrap)
        return ntuple(i->fwrap(i,x), ϕ.num_tilings)
    else
        freg(i,x) = get_tiling(ϕ.bins, ϕ.lidxs, ϕ.num_inputs, x, offsets[i], tile_index)
        return ntuple(i->freg(i,x), ϕ.num_tilings)
    end
end

function (ϕ::TileCodingBasis{<:Vector{<:StepRangeLen}})(x)
    offset(i) = ntuple(j->(i-1)/ϕ.num_tilings * (1/length(ϕ.bins[j])), ϕ.num_inputs)
    if ϕ.tiling_type == :wrap
        fwrap(i,x) = get_tiling(ϕ.bins, ϕ.lidxs, ϕ.num_inputs, x, offset(i), tile_index_wrap)
        return ntuple(i->fwrap(i,x), ϕ.num_tilings)
    else
        freg(i,x) = get_tiling(ϕ.bins, ϕ.lidxs, ϕ.num_inputs, x, offsets(i), tile_index)
        return ntuple(i->freg(i,x), ϕ.num_tilings)
    end
end

function (ϕ::TileCodingBasis{<:AbstractArray{T,3}})(x) where {T}
    if ϕ.tiling_type == :wrap
        fwrap(i,x) = get_tiling(view(ϕ.bins,:,:,i), ϕ.lidxs, ϕ.num_inputs, x, tile_index_wrap)
        return ntuple(i->fwrap(i,x), ϕ.num_tilings)
    else
        freg(i,x) = get_tiling(view(ϕ.bins,:,:,i), ϕ.lidxs, ϕ.num_inputs, x, tile_index)
        return ntuple(i->freg(i,x), ϕ.num_tilings)
    end
end

function (ϕ::TileCodingBasis{Vector{Vector{Vector{T}}}})(x) where {T}
    if ϕ.tiling_type == :wrap
        fwrap(i,x) = get_tiling(ϕ.bins[i], ϕ.lidxs, ϕ.num_inputs, x, tile_index_wrap)
        return ntuple(i->fwrap(i,x), ϕ.num_tilings)
    else
        freg(i,x) = get_tiling(ϕ.bins[i], ϕ.lidxs, ϕ.num_inputs, x, tile_index)
        return ntuple(i->freg(i,x), ϕ.num_tilings)
    end
end


"""
    length(ϕ::TileCodingBasis)

Returns the number of feautes produced by the tile coding basis. 
"""

function Base.length(ϕ::TileCodingBasis{<:StepRangeLen})
    n = length(ϕ.bins)
    return n^ϕ.num_inputs * ϕ.num_tilings
end

function Base.length(ϕ::TileCodingBasis{<:Vector{<:StepRangeLen}})
    return prod(length, ϕ.bins) * ϕ.num_tilings
end

function Base.length(ϕ::TileCodingBasis{<:AbstractArray{T,3}}) where {T}
    n,m,k = size(ϕ.bins)
    return n^m * k
end

function Base.length(ϕ::TileCodingBasis{Vector{Vector{Vector{T}}}}) where {T}
    return prod(length, ϕ.bins[1]) * length(ϕ.bins)
end

"""
    size(ϕ::TileCodingBasis)

Returns the number of tiles and tilings (num_tiles_per_tiling, num_tilings) for the tile coding basis.
"""
function Base.size(ϕ::TileCodingBasis{<:StepRangeLen})
    n = length(ϕ.bins)
    return (n^ϕ.num_inputs,  ϕ.num_tilings)
end

function Base.size(ϕ::TileCodingBasis{<:Vector{<:StepRangeLen}})
    return (prod(length, ϕ.bins), ϕ.num_tilings)
end

function Base.size(ϕ::TileCodingBasis{<:AbstractArray{T,3}}) where {T}
    n,m,k = size(ϕ.bins)
    return (n^m,k)
end

function Base.size(ϕ::TileCodingBasis{Vector{Vector{Vector{T}}}}) where {T}
    return prod(length, ϕ.bins[1]), length(ϕ.bins)
end