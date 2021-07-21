module DecisionMakingUtils

using Base: Float64
using LinearAlgebra
using OnlineStats
import OnlineStats: fit!
import Base: length, eltype

export FourierBasis, FourierBasisBuffer, ConcatentateBasis
export LinearNormalization, ZeroOneNormalization, PosNegNormalization, GaussianNormalization
export gaussian_stats, extrema_stats, fit!
export BufferedFunction
export update!

include("basis/basis.jl")
include("bufferedfun.jl")
# include("normalization.jl")
include("normalization_constant.jl")
include("stats.jl")


end
