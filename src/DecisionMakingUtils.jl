module DecisionMakingUtils

using Base: Float64
using LinearAlgebra
using OnlineStats
using EllipsisNotation
using ChainRulesCore

import OnlineStats: fit!
import Base: length, eltype, size


import ChainRulesCore: rrule, Tangent
export FourierBasis, FourierBasisBuffer, ConcatentateBasis, TileCodingBasis
export LinearNormalization, ZeroOneNormalization, PosNegNormalization, GaussianNormalization
export gaussian_stats, extrema_stats, fit!
export BufferedFunction
export update!
export TileCodingModel, LinearBuffer, TabularModel, LinearModel
export value_withgrad

include("basis/basis.jl")
include("bufferedfun.jl")
include("func_approx/functionapprox.jl")
# include("normalization.jl")
include("normalization_constant.jl")
include("stats.jl")


end
