using DecisionMakingUtils
using Test

@testset "TileCodingModel Test" begin
    num_tiles = 3
    num_tilings = 1
    num_actions = 1
    num_outputs = 1
    dims = 1
    tcconf = TileCoderConfig(num_tiles, num_tilings, dims; offset = "cascade", scale_output = false, input_ranges = nothing, bound = "clip")
    ϕ = TC(tcconf)
    m = TileCodingModel(ϕ, num_outputs, num_tiles, num_tilings, num_actions)

    s = 0.0
    t = ϕ(s)
    m.w[1, 1, 1, 1] = 10.0
    y = m(s)
    @test y == 10.0
end