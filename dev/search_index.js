var documenterSearchIndex = {"docs":
[{"location":"#DecisionMakingUtils.jl","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.jl","text":"","category":"section"},{"location":"","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.jl","text":"Documentation for DecisionMakingUtils.jl","category":"page"},{"location":"","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.jl","text":"Modules = [DecisionMakingUtils]","category":"page"},{"location":"#DecisionMakingUtils.FourierBasis","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.FourierBasis","text":"FourierBasis([::Type,] num_inputs::Int, dorder::Int, iorder::Int [, full::Bool=false])\n\nCreates a struct to generate fourier features up to a given order. Both coupled fourier features, e.g.,  cos(π(3x₁ + 2x₂)) and uncoupled features, cos(πx₁) cos(π2x₁)  can be generated with this basis function.  The dorder parameter controls order for the coupling features. The number of coupled features generated, (textdorder+1)^textnum_inputs,  grows exponentially with dorder, so it is not reccomended for use with high deminsional vectors. The iorder parameter controls the order of the independent features generated. The full parameter determines if both  sin and cos features are generated, if false only cos features are generated. \n\nSee also: FourierBasisBuffer\n\nExamples\n\njulia> f = FourierBasis(2, 1, 2);\n\njulia> x = [0.0, 0.5];\n\njulia> feats = f(x)\n6-element Array{Float64,1}:\n  1.0\n  6.123233995736766e-17\n  1.0\n  6.123233995736766e-17\n  1.0\n -1.0\n\n\n\n\n\n\n","category":"type"},{"location":"#DecisionMakingUtils.FourierBasisBuffer","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.FourierBasisBuffer","text":"FourierBasisBuffer(ϕ::FourierBasis)\n\nCreates preallocated buffers for a fourier basis to use to avoid making allocations on every call of the basis. \n\nSee also: FourierBasis\n\nExamples\n\n```jldoctest julia> f = FourierBasis(2, 1, 2);\n\njulia> buff = FourierBasisBuffer(f);\n\njulia> x = [0.0, 0.5];\n\njulia> feats = f(buff, x) 6-element Array{Float64,1}:   1.0   6.123233995736766e-17   1.0   6.123233995736766e-17   1.0  -1.0\n\n\n\n\n\n","category":"type"},{"location":"#DecisionMakingUtils.LinearNormalization","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.LinearNormalization","text":"LinearNormalization{T}(a::T,b::T)\n\nThis is a functor that normalizes a vector x as (x - a) * b. This is the standard interface for all  linear normalizations such as mapping to 01, -11 and mean zero standard deviation one.  LinearNormalization also supports the functions Base.lenght and Base.eltype.\n\nSee also: PosNegNormalization, GaussianNormalization\n\nExamples\n\njulia> nrm = LinearNormalization([0.1, 2.0], [1.0, 0.5]);\n\njulia> x = [1.0, 2.0];\n\njulia> nrm(x)\n2-element Vector{Float64}:\n 0.9\n 0.0\n\njulia> nrm = LinearNormalization(2);  # no scaling to input vector\n\njulia> nrm(x)\n2-element Vector{Float64}:\n 1.0\n 2.0\n\njulia> low = [0.0, -1.0];\n\njulia> high = [3.0, 0.5];\n\njulia> nrm = ZeroOneNormalization(low, high);  # normalize each entry to [0,1]\n\njulia> x = [1.0, 0.0];\n\njulia> feats = nrm(x)\n2-element Array{Float64,1}:\n0.3333333333333333\n0.6666666666666666\n\njulia> y = zero(x);  # create buffer to prevent allocations\n\njulia> feats = nrm(y, x);  # no allocation return\n\njulia> nrm = PosNegNormalization(low, high);  # normalize each entry to [0,1]\n\njulia> nrm(x)\n2-element Vector{Float64}:\n -0.3333333333333333\n  0.3333333333333333\n\njulia> μ = [0.0, 1.0];  # vector of means\n\njulia> σ = [1.0, 2.0];  # vector of standard deviations\n\njulia> nrm = GaussianNormalization(μ, σ);  # normalize x to be mean 0 and standard deviation 1\n\njulia> nrm(x)\n2-element Vector{Float64}:\n  1.0\n -0.5\n\n\n\n\n\n","category":"type"},{"location":"#DecisionMakingUtils.TileCodingBasis","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.TileCodingBasis","text":"TileCodingBasis([::Type,] num_inputs::Int, num_tiles::Int)\n\nCreates a struct to generate fourier features up to a given order. Both coupled fourier features, e.g.,  cos(π(3x₁ + 2x₂)) and uncoupled features, cos(πx₁) cos(π2x₁)  can be generated with this basis function.  The dorder parameter controls order for the coupling features. The number of coupled features generated, (textdorder+1)^textnum_inputs,  grows exponentially with dorder, so it is not reccomended for use with high deminsional vectors. The iorder parameter controls the order of the independent features generated. The full parameter determines if both  sin and cos features are generated, if false only cos features are generated. \n\nSee also: FourierBasisBuffer\n\nExamples\n\njulia> f = TileCodingBasis(2, 3);\n\njulia> x = [0.0, 0.5];\n\njulia> feats = f(x)\n6-element Array{Float64,1}:\n  1.0\n  6.123233995736766e-17\n  1.0\n  6.123233995736766e-17\n  1.0\n -1.0\n\n\n\n\n\n\n","category":"type"},{"location":"#Base.length-Union{Tuple{FourierBasis{T,false}}, Tuple{T}} where T","page":"DecisionMakingUtils.jl","title":"Base.length","text":"length(ϕ::FourierBasis)\n\nReturns the number of feautes produced by the Fourier basis. \n\n\n\n\n\n","category":"method"},{"location":"#DecisionMakingUtils.extrema_stats-Union{Tuple{T}, Tuple{Type{T},Int64}} where T","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.extrema_stats","text":"extrema_stats([::Type{T},] num_features::Int)\n\nThis function creates an OnlineStats.KahanVariance object for tracking the mean and variance for a vector. Any OnlineStats.Weight can be used. The default is OnlineStats.EqualWeight and OnlineStats.ExponentialWeight  if an integer or float is given as the weight. \n\nSee also: extrema_stats, LinearNormalization\n\nExamples\n\njulia> stats = gaussian_stats(Float32, 2, 1e-4)\nGroup\n├─ KahanVariance: n=0 | value=1.0\n└─ KahanVariance: n=0 | value=1.0\n\njulia> fit!(stats, [1.0, 2.0])\nGroup\n├─ KahanVariance: n=1 | value=1.0\n└─ KahanVariance: n=1 | value=1.0\n\n\n\n\n\n\n","category":"method"},{"location":"#DecisionMakingUtils.gaussian_stats","page":"DecisionMakingUtils.jl","title":"DecisionMakingUtils.gaussian_stats","text":"gaussian_stats([::Type{T},] num_features::Int[, weight])\n\nThis function creates an OnlineStats.KahanVariance object for tracking the mean and variance for a vector. Any OnlineStats.Weight can be used. The default is OnlineStats.EqualWeight and OnlineStats.ExponentialWeight  if an integer or float is given as the weight. \n\nSee also: extrema_stats, LinearNormalization\n\nExamples\n\njulia> stats = gaussian_stats(Float32, 2, 1e-4)\nGroup\n├─ KahanVariance: n=0 | value=1.0\n└─ KahanVariance: n=0 | value=1.0\n\njulia> fit!(stats, [1.0, 2.0])\nGroup\n├─ KahanVariance: n=1 | value=1.0\n└─ KahanVariance: n=1 | value=1.0\n\n\n\n\n\n\n","category":"function"}]
}
