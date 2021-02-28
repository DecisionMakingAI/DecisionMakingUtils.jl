using DecisionMakingUtils
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
    println(low)
    @test all(y .≥ -1.0)
    @test all(y .≤ 1.0)
    y = n(high)
    @test all(y .≥ -1.0)
    @test all(y .≤ 1.0)
    n(buff, high)
    @test all(isapprox.(buff, y))
end

@testset "GaussianNormalization Test" begin
    k = 3
    T = Float32
    μ = collect(T, 1:k)
    σ = collect(T, 1:k)
    n = GaussianNormalization(μ, σ)
    
    x = zeros(T, k)
    buff = zero(x)
    y = n(x)
    @test typeof(y) == typeof(x)
    @test size(y) == size(x)
    println(y)
    @test all(isapprox.(y, -1.0))
    y = n(μ)
    @test all(isapprox.(y, 0.0))
    n(buff, μ)
    @test all(isapprox.(buff, y))
end