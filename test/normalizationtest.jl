using DecisionMakingUtils
using OnlineStats
using Test

@testset "ZeroOneNormalization Test" begin
    k = 3
    T = Float32
    low = -collect(T, 1:k)
    high = collect(T, 1:k)
    ranges = hcat(low, high)
    n = ZeroOneNormalization(ranges)
    x = zeros(T, k)
    buff = zero(x)
    y = n(x)
    @test typeof(y) == typeof(x)
    @test size(y) == size(x)

    @test all(y .≥ 0.0)
    @test all(y .≤ 1.0)
    y = n(low)
    @test all(y .≥ 0.0)
    @test all(y .≤ 1.0)
    y = n(high)
    @test all(y .≥ 0.0)
    @test all(y .≤ 1.0)
    n(buff, high)
    @test all(isapprox.(buff, y))

    # @. low -= 1
    # @. high += 1
    # n(buff, low)
    # @test all(buff .< 0.0)
    # n(buff, low, fit=true)
    # @test all(buff .≥ 0.0)
    # n(buff, high)
    # @test all(buff .> 1.0)
    # n(buff, high, fit=true)
    # @test all(buff .≤ 1.0)
end

@testset "PosNegNormalization Test" begin
    k = 3
    T = Float32
    low = -collect(T, 1:k)
    high = collect(T, 1:k)
    ranges = hcat(low, high)
    n = PosNegNormalization(ranges)

    x = zeros(T, k)
    buff = zero(x)
    y = n(x)
    @test typeof(y) == typeof(x)
    @test size(y) == size(x)
    @test all(y .≥ -1.0)
    @test all(y .≤ 1.0)
    y = n(low)
    
    @test all(y .≥ -1.0)
    @test all(y .≤ 1.0)
    y = n(high)
    @test all(y .≥ -1.0)
    @test all(y .≤ 1.0)
    n(buff, high)
    @test all(isapprox.(buff, y))

    # TODO determine if we want to support adaptable basis range
    # @. low -= 1
    # @. high += 1
    # n(buff, low)
    # @test all(buff .< -1.0)
    # n(buff, low, fit=true)
    # @test all(buff .≥ -1.0)
    # n(buff, high)
    # @test all(buff .> 1.0)
    # n(buff, high, fit=true)
    # @test all(buff .≤ 1.0)
end

@testset "GaussianNormalization Test" begin
    k = 3
    T = Float32
    x1 = collect(T, 1:k)
    x2 = collect(T, 1:k) .* 2 
    x3 = collect(T, 1:k) .* 3
    
    n = GaussianNormalization(k)
    
    x = zeros(T, k)
    buff = zero(x)
    y = n(x)
    @test typeof(y) == typeof(x)
    @test size(y) == size(x)
    
    @test all(isapprox.(y, 0.0))
    @. x = 1
    y = n(x)
    @test all(isapprox.(y, 1.0))
    
    n(buff, x)
    @test all(isapprox.(buff, y))

    v = k*KahanVariance()
    x1 = collect(T, 1:k)
    x2 = collect(T, 1:k) .* 2 
    x3 = collect(T, 1:k) .* 3

    fit!(v, x1)
    @. y = (x1 - mean.(v.stats)) / std.(v.stats)
    # n(buff, x1, fit=true)
    # @test all(isapprox(buff, y))

    fit!(v, x2)
    @. y = (x2 - mean.(v.stats)) / std.(v.stats)
    # n(buff, x2, fit=true)
    # @test all(isapprox(buff, y))

    fit!(v, x3)
    @. y = (x3 - mean(v.stats)) / std(v.stats)
    # n(buff, x3, fit=true)
    # @test all(isapprox(buff, y))
end