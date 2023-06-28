using DecisionMakingUtils
using Test

@testset "TileCodingModel Test" begin
    # test single action, single tiling, single output
    num_tiles = 3
    num_tilings = 1
    num_actions = 1
    num_outputs = 1
    dims = 1
    ϕ = TileCodingBasis(dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)

    vec(m.w) .= 1:length(m.w)
    @test m(0.0) == 1
    @test m(0.0,1) == 1
    @test m(1.0) == 1
    @test m(0.99) == 3
    @test m(0.5) == 2
    @test m(0.5,1) == 2
    grad = zero(m.w)
    grad[1,1,1,1] = 1
    v, g = value_withgrad(m, 0.0)
    @test v == 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.0, 1)
    @test v == 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 1.0)
    @test v == 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.99)
    @test v == 3
    @. grad = 0.0
    grad[1,3,1,1] = 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.5)
    @test v == 2
    @. grad = 0.0
    grad[1,2,1,1] = 1
    @test all(isapprox.(g, grad))

    @test_throws "Not a valid action" m(0.0,2)
    @test_throws "Not a valid action" value_withgrad(m,0.0,4)
    @test_throws AssertionError m(0.0, -1)
    @test_throws AssertionError value_withgrad(m, 0.0,-1)

    # test single action with single tiling and single output
    num_actions = 3
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)
    @test m(0.0) == [1.0, 4.0, 7.0]
    @test m(1.0) == [1.0, 4.0, 7.0]
    @test m(0.99) == [3.0, 6.0, 9.0]
    @test m(0.5) == [2.0, 5.0, 8.0]
    @test m(0.0,1) == 1.0 
    @test m(0.0,2) == 4.0 
    @test m(0.99,2) == 6.0 
    @test m(0.5,3) == 8.0
    @test_throws "Not a valid action" m(0.0,4)
    @test_throws "Not a valid action" value_withgrad(m, 0.0,4)
    @test_throws AssertionError m(0.0,-3)
    @test_throws AssertionError value_withgrad(m, 0.0,10)
    grad = zero(m.w)

    grad[1,1,1,1] = 1
    v, g = value_withgrad(m, 0.0, 1)
    @test v == 1.0
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 1.0, 3)
    @test v == 7.0
    @. grad = 0.0
    grad[1,1,1,3] = 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.99, 2)
    @test v == 6.0
    @. grad = 0.0
    grad[1,3,1,2] = 1
    v, g = value_withgrad(m, 0.5)
    @test v == [2.0, 5.0, 8.0]
    @. grad = 0.0
    grad[1,2,1,:] .= 1
    @test all(isapprox.(g, grad))

    # test multiple tilings with single action
    num_tilings = 2
    num_actions = 1
    ϕ = TileCodingBasis(dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)
    @test m(0.0) == 1 + 4
    @test m(1.0) == 1 + 4
    @test m(0.99) == 3 + 4
    @test m(0.67) == 3 + 6
    @test m(0.66) == 2 + 6
    @test m(0.33) == 1 + 5
    
    grad = zero(m.w)
    v, g = value_withgrad(m, 0.0)
    @test v == 1 + 4
    @. grad = 0.0
    grad[1,1,1,1] = 1
    grad[1,1,2,1] = 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.66)
    @test v == 2 + 6
    @. grad = 0.0
    grad[1,2,1,1] = 1
    grad[1,3,2,1] = 1
    @test all(isapprox.(g, grad))

    @test_throws "Not a valid action" m(0.0,2)
    @test_throws "Not a valid action" value_withgrad(m, 0.0,4)

    # test multiple actions with multiple tiles
    num_actions = 2
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)
    @test m(0.0) == [1.0 + 4.0, 7.0 + 10.0]
    @test m(1.0) == [1.0 + 4.0, 7.0 + 10.0]
    @test m(0.99) == [3.0 + 4.0, 9.0 + 10.0]
    @test m(0.66) == [2.0 + 6.0, 8.0 + 12.0]
    @test m(0.33) == [1.0 + 5.0, 7.0 + 11.0]
    @test m(0.0,1) == 1.0 + 4.0
    @test m(0.0,2) == 7.0 + 10.0
    @test m(0.99,2) == 9.0 + 10.0
    @test m(0.66,1) == 2.0 + 6.0
    @test m(0.33,2) == 7.0 + 11.0

    grad = zero(m.w)
    v, g = value_withgrad(m, 0.0, 1)
    @test v == 1.0 + 4.0
    @. grad = 0.0
    grad[1,1,1,1] = 1
    grad[1,1,2,1] = 1
    @test all(isapprox.(g, grad))

    v, g = value_withgrad(m, 0.66, 2)
    @test v == 8.0 + 12.0
    @. grad = 0.0
    grad[1,2,1,2] = 1
    grad[1,3,2,2] = 1
    @test all(isapprox.(g, grad))

    @test_throws "Not a valid action" m(0.0,4)
    @test_throws "Not a valid action" value_withgrad(m, 0.0,-3)
    @test_throws AssertionError m(0.0,-1)
    @test_throws AssertionError value_withgrad(m, 0.0,0)

    # test multi outputs with single action and single tiling
    num_outputs = 2
    num_tilings = 1
    num_actions = 1
    ϕ = TileCodingBasis(dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)

    @test m(0.0) == [1.0, 2.0]
    @test m(1.0) == [1.0, 2.0]
    @test m(0.99) == [5.0, 6.0]
    @test m(0.66) == [3.0, 4.0]
    
    grad = zero(m.w)
    v, g = value_withgrad(m, 0.0)
    @test v == [1.0, 2.0]
    @. grad = 0.0
    grad[1,1,1,1] = 1
    grad[2,1,1,1] = 1
    @test all(isapprox.(g, grad))

    v, g = value_withgrad(m, 0.66)
    @test v == [3.0, 4.0]
    @. grad = 0.0
    grad[1,2,1,1] = 1
    grad[2,2,1,1] = 1
    @test all(isapprox.(g, grad))

    # test multi outputs with single action and multiple tilings
    num_tilings = 2
    num_actions = 1
    ϕ = TileCodingBasis(dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)

    @test m(0.0) == [1.0 + 7.0, 2.0 + 8.0]
    @test m(1.0) == [1.0 + 7.0, 2.0 + 8.0]
    @test m(0.99) == [5.0 + 7.0, 6.0 + 8.0]
    @test m(0.66) == [3.0 + 11.0, 4.0 + 12.0]
    @test m(0.34) == [3.0 + 9.0, 4.0 + 10.0]
    @test m(0.67) == [5.0 + 11.0, 6.0 + 12.0]

    grad = zero(m.w)
    v, g = value_withgrad(m, 0.0)
    @test v == [1.0 + 7.0, 2.0 + 8.0]
    @. grad = 0.0
    grad[:,1,:,1] .= 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.0, 1)
    @test v == [1.0 + 7.0, 2.0 + 8.0]
    @test all(isapprox.(g, grad))

    v, g = value_withgrad(m, 0.66)
    @test v == [3.0 + 11.0, 4.0 + 12.0]
    @. grad = 0.0
    grad[:,2,1,1] .= 1
    grad[:,3,2,1] .= 1
    @test all(isapprox.(g, grad))

    # test multi outputs with multi actions and multi tilings
    num_actions = 2
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    vec(m.w) .= 1:length(m.w)

    @test m(0.0) == reshape([1.0 + 7.0, 2.0 + 8.0, 13.0 + 19.0, 14.0 + 20.0], (2,2))
    @test m(0.0,1) == [1.0 + 7.0, 2.0 + 8.0]
    @test m(0.0,2) == [13.0 + 19.0, 14.0 + 20.0]
    @test m(0.99) == reshape([5.0 + 7.0, 6.0 + 8.0, 17.0 + 19.0, 18.0 + 20.0], (2,2))
    @test m(0.66) == reshape([3.0 + 11.0, 4.0 + 12.0, 15.0 + 23.0, 16.0 + 24.0], (2,2))
    @test m(0.34) == reshape([3.0 + 9.0, 4.0 + 10.0, 15.0 + 21.0, 16.0 + 22.0], (2,2))

    grad = zero(m.w)
    v, g = value_withgrad(m, 0.0)
    @test v == reshape([1.0 + 7.0, 2.0 + 8.0, 13.0 + 19.0, 14.0 + 20.0], (2,2))
    @. grad = 0.0
    grad[:,1,:,:] .= 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.0, 1)
    @test v == [1.0 + 7.0, 2.0 + 8.0]
    @. grad[:,:,:,2] = 0
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.0, 2)
    @test v == [13.0 + 19.0, 14.0 + 20.0]
    @. grad[:,:,:,1] = 0
    @. grad[:,1,:,2] .= 1
    @test all(isapprox.(g, grad))

    v, g = value_withgrad(m, 0.66)
    @test v == reshape([3.0 + 11.0, 4.0 + 12.0, 15.0 + 23.0, 16.0 + 24.0], (2,2))
    @. grad = 0.0
    grad[:,2,1,:] .= 1
    grad[:,3,2,:] .= 1
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.66, 1)
    @test v == [3.0 + 11.0, 4.0 + 12.0]
    @. grad[:,:,:,2] = 0
    @test all(isapprox.(g, grad))
    v, g = value_withgrad(m, 0.66, 2)
    @test v == [15.0 + 23.0, 16.0 + 24.0]
    @. grad[:,:,:,1] = 0
    @. grad[:,2,1,2] .= 1
    @. grad[:,3,2,2] .= 1
    @test all(isapprox.(g, grad))

end



@testset "TileCodingModel With Buffer Test" begin
    num_tiles = 3
    num_tilings = 3
    num_actions = 4
    num_outputs = 2
    dims = 1
    ϕ = TileCodingBasis(dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
    m = TileCodingModel(ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
    buff = LinearBuffer(m)
    vec(m.w) .= 1:length(m.w)

    bf = BufferedFunction(m, buff)

    v1 = m(0.0)
    v2 = m(buff, 0.0)
    bc = copy(buff.output)
    bo = copy(v2)
    v3 = bf(0.0)
    @test v1 == v2
    @test v1 == v3
    @test all(bc .== buff.output)
    @test all(bo .== v1)

    v1 = m(0.6, 1)
    v2 = m(buff, 0.6, 1)
    bc = copy(buff.output)
    bo = copy(v2)
    v3 = bf(0.6, 1)
    @test v1 == v2
    @test v1 == v3
    @test all(bc .== buff.output)
    @test all(bo .== v1)

    v1, g1 = value_withgrad(m, 0.6, 1)
    v2, g2 = value_withgrad(buff, m, 0.6, 1)
    bc = copy(buff.output)
    bo = copy(v2)
    bg = copy(buff.grad)
    v3, g3 = value_withgrad(bf, 0.6, 1)
    @test v1 == v2
    @test v1 == v3
    @test all(bc .== buff.output)
    @test all(bo .== v1)
    @test all(bg .== buff.grad)
    @test all(bg .== g1)

    v1, g1 = value_withgrad(m, 0.95)
    v2, g2 = value_withgrad(buff, m, 0.95)
    bc = copy(buff.output)
    bo = copy(v2)
    bg = copy(buff.grad)
    v3, g3 = value_withgrad(bf, 0.95)
    @test v1 == v2
    @test v1 == v3
    @test all(bc .== buff.output)
    @test all(bo .== v1)
    @test all(bg .== buff.grad)
    @test all(bg .== g1)

end