module DecisionMakingUtils

using LinearAlgebra
using OnlineStats
import Base: length

export FourierBasis, FourierBasisBuffer, ConcatentateBasis
export ZeroOneNormalization, PosNegNormalization, GaussianNormalization

include("basis/basis.jl")
include("normalization.jl")

end
