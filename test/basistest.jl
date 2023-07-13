using DecisionMakingUtils
using Test

@testset "TileCodingBasis Test" begin
    num_tiles = 3
    num_tilings = 1
    num_inputs = 1
    tiling_type = :wrap
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test typeof(f(0.0)) == Tuple{Int}
    @test length(f) == 3
    @test size(f) == (3,1)
    @test f(-1.0) == (1,)
    @test f(0.0) == (1,)
    @test f(0.1) == (1,)
    @test f(0.34) == (2,)
    @test f(0.35) == (2,)
    @test f(0.66) == (2,)
    @test f(0.67) == (3,)
    @test f(0.99) == (3,)
    @test f(1.0) == (1,)
    @test f([0.0]) == (1,)
    @test f([0.1]) == (1,)
    @test f([1.0]) == (1,)
    
    tiling_type = :clip
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test f(1.0) == (3,)
    @test f(1.1) == (3,)
    tiling_type = :wrap
    num_tilings = 2
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test isa(f(0.0), Tuple{Int,Int})
    @test length(f) == 6
    @test size(f) == (num_tiles,num_tilings)
    @test f(0.0) == (1,1)
    @test f(0.16) == (1,1)
    @test f(0.17) == (1,2)
    @test f(0.34) == (2,2)
    @test f(0.66) == (2,3)
    @test f(0.67) == (3,3)
    @test f(0.99) == (3,1)
    @test f(1.0) == (1,1)
    
    tiling_type = :clip
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test f(0.0) == (1,1)
    @test f(0.16) == (1,1)
    @test f(0.17) == (1,2)
    @test f(0.34) == (2,2)
    @test f(0.66) == (2,3)
    @test f(0.67) == (3,3)
    @test f(0.99) == (3,3)
    @test f(1.0) == (3,3)
    
    num_tilings = 4
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test isa(f(0.0), NTuple{num_tilings,Int})
    @test length(f) == 12
    @test size(f) == (num_tiles,num_tilings)
    @test f(0.0) == (1,1,1,1)
    @test f(0.09) == (1,1,1,2)
    @test f(0.174) == (1,1,2,2)
    @test f(0.257) == (1,2,2,2)
    @test f(0.34) == (2,2,2,2)
    @test f(0.66) == (2,3,3,3)
    @test f(0.67) == (3,3,3,3)
    @test f(0.99) == (3,3,3,3)
    @test f(1.0) == (3,3,3,3)

    tiling_type = :wrap
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test f(0.0) == (1,1,1,1)
    @test f(0.916) == (3,3,1,1)
    @test f(0.917) == (3,1,1,1)
    @test f(0.99) == (3,1,1,1)
    @test f(1.0) == (1,1,1,1)

    num_inputs = 2
    num_tilings = 1
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test length(f) == (num_tiles^num_inputs) * num_tilings
    @test size(f) == (num_tiles^num_inputs,num_tilings)
    @test f([0.0, 0.0]) == (1,)
    @test f([0.0, 0.1]) == (1,)
    @test f([0.0, 0.34]) == (4,)
    @test f([0.34, 0.1]) == (2,)
    @test f([0.34, 0.34]) == (5,)
    @test f([0.0, 0.99]) == (7,)
    @test f([0.0, 1.0]) == (1,)
    @test f([0.99, 1.0]) == (3,)
    @test f([1.0, 1.0]) == (1,)
    @test f([1.0, 0.0]) == (1,)
    @test f([0.99, 0.0]) == (3,)
    @test f([0.99, 0.99]) == (9,)

    num_tilings = 2
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type)
    @test length(f) == (num_tiles^num_inputs) * num_tilings
    @test size(f) == (num_tiles^num_inputs,num_tilings)
    @test f([0.0, 0.0]) == (1,1)
    @test f([0.0, 0.1]) == (1,1)
    @test f([0.0, 0.33]) == (1,4)
    @test f([0.0, 0.34]) == (4,4)
    @test f([0.33, 0.1]) == (1,2)
    @test f([0.34, 0.1]) == (2,2)
    @test f([0.34, 0.33]) == (2,5)
    @test f([0.34, 0.34]) == (5,5)
    @test f([0.0, 0.99]) == (7,1)
    @test f([0.0, 1.0]) == (1,1)
    @test f([0.99, 1.0]) == (3,1)
    @test f([1.0, 1.0]) == (1,1)
    @test f([1.0, 0.0]) == (1,1)
    @test f([0.99, 0.99]) == (9,1)

    num_tilings = 1
    tilings_per_dim = [3,]
    f = TileCodingBasis(tilings_per_dim, num_tilings=num_tilings, tiling_type=tiling_type)
    @test typeof(f(0.0)) == Tuple{Int}
    @test length(f) == 3
    @test size(f) == (3,1)
    @test f(0.0) == (1,)
    @test f(0.1) == (1,)
    @test f(0.34) == (2,)
    @test f(0.35) == (2,)
    @test f(0.66) == (2,)
    @test f(0.67) == (3,)
    @test f(0.99) == (3,)
    @test f(1.0) == (1,)
    @test f([0.0]) == (1,)
    @test f([0.1]) == (1,)
    @test f([1.0]) == (1,)

    tiles_per_dim = [3,2,4]
    f = TileCodingBasis(tiles_per_dim, num_tilings=num_tilings, tiling_type=tiling_type)
    @test typeof(f([0.0, 0.0, 0.0])) == Tuple{Int}
    @test length(f) == prod(tiles_per_dim) * num_tilings
    @test size(f) == (prod(tiles_per_dim),1)
    @test f([0.0, 0.0, 0.0]) == (1,)
    @test f([1.0, 1.0, 1.0]) == (1,)
    @test f([0.99, 0.99, 0.99]) == (prod(tiles_per_dim),)

    num_tilings = 2
    f = TileCodingBasis(tiles_per_dim, num_tilings=num_tilings, tiling_type=tiling_type)
    @test typeof(f([0.0, 0.0, 0.0])) == Tuple{Int,Int}
    @test length(f) == prod(tiles_per_dim) * num_tilings
    @test size(f) == (prod(tiles_per_dim),num_tilings)
    @test f([0.0, 0.0, 0.0]) == (1,1)
    @test f([1.0, 1.0, 1.0]) == (1,1)
    @test f([0.99, 0.99, 0.99]) == (prod(tiles_per_dim),1)

    num_tilings = 2
    f = TileCodingBasis(tiles_per_dim, num_tilings=num_tilings, tiling_type=:clip)
    @test typeof(f([0.0, 0.0, 0.0])) == Tuple{Int,Int}
    @test length(f) == prod(tiles_per_dim) * num_tilings
    @test size(f) == (prod(tiles_per_dim),num_tilings)
    @test f([0.0, 0.0, 0.0]) == (1,1)
    @test f([1.0, 1.0, 1.0]) == (prod(tiles_per_dim),prod(tiles_per_dim))
    @test f([0.99, 0.99, 0.99]) == (prod(tiles_per_dim),prod(tiles_per_dim))

    num_tilings = 4
    f = TileCodingBasis(tiles_per_dim, num_tilings=num_tilings, tiling_type=tiling_type, tile_loc=:random)
    @test typeof(f([0.0, 0.0, 0.0])) == NTuple{num_tilings,Int}
    @test length(f) == prod(tiles_per_dim) * num_tilings
    @test size(f) == (prod(tiles_per_dim),num_tilings)
    @test f([0.0, 0.0, 0.0]) == (1,1,1,1)
    
    num_tilings = 5
    f = TileCodingBasis(tiles_per_dim, num_tilings=num_tilings, tiling_type=tiling_type, tile_loc=:random)
    @test typeof(f([0.0, 0.0, 0.0])) == NTuple{num_tilings,Int}
    @test length(f) == prod(tiles_per_dim) * num_tilings
    @test size(f) == (prod(tiles_per_dim),num_tilings)
    @test f([0.0, 0.0, 0.0]) == (1,1,1,1,1)

    num_tilings = 10
    f = TileCodingBasis(num_inputs, num_tiles, num_tilings=num_tilings, tiling_type=tiling_type, tile_loc=:random)
    @test typeof(f([0.0, 0.0, 0.0])) == NTuple{num_tilings,Int}
    @test length(f) == (num_tiles^num_inputs) * num_tilings
    @test size(f) == (num_tiles^num_inputs,num_tilings)
end