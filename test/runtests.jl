using SafeTestsets

@safetestset "Normalization Test Sets" begin include("normalizationtest.jl") end
@safetestset "Basis Test Sets" begin include("basistest.jl") end
# @safetestset "Function Approx Test Sets" begin include("functionapproxtest.jl") end

