module DecisionMakingUtils

using LinearAlgebra
using OnlineStats
import Base: length

export FourierBasis, FourierBasisBuffer, ConcatentateBasis
export ZeroOneNormalization, PosNegNormalization, GaussianNormalization
export BufferedFunction

include("basis/basis.jl")
include("normalization.jl")
include("bufferedfun.jl")

end
