# DecisionMakingUtils

<!--[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingUtils.jl/stable)-->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingUtils.jl/dev)
[![Build Status](https://github.com/DecisionMakingAI/DecisionMakingUtils.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/DecisionMakingUtils.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/DecisionMakingUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/DecisionMakingUtils.jl)


This package contains utility functions used through other DecisionMakingAI repositories. Currently there is functionality for creating a Fourier basis, Tile Coding, normalizing features, and linear alue function modeling. 

Example creating a tile coding based q function. 
```julia
using DecisionMakingUtils
using Flux: Chain

X = [1.0 2.0; -3.0 4.0]  # Assume X represents the ranges of the state features where the first (second) column represents the minimum (maximum).  
state_dims = size(X, 1)
num_tiles = 5
num_tilings = 4
num_actions = 4

nrm = ZeroOneNormalization(X)  # TileCoding assumes the features are normalized to [0,1]. Wrapping tiles will make features >=1 wrap around starting from 0
nbuff = zeros(state_dims)  # Buffer to prevent allocations of the feature normalization
nf = BufferedFunction(nrm, nbuff)  # wrapper function to hold the buffer
tc = TileCodingBasis(state_dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
ϕ = Chain(nf, tc)  # chain the normalization and tile coding into one function. 

num_outputs = 1 # if you want to predict successor features, this should be length(tc)
m = TileCodingModel(ϕ, num_tiles=size(tc)[1], num_tilings=num_tilings, num_outputs=num_outputs,num_actions=num_actions)
buff = LinearBuffer(m)
bf = BufferedFunction(m, buff)
s = rand(state_dims)
qs = bf(s) # value of each q function
qsa = bf(s, 1) # value of first action in state s
qs, grad = value_withgrad(bf, s) # same as qs above, plus the gradient w.r.t. each action this is just phi(s) for each a. grad has shape of m.w (model weights)
qsa, grad = value_withgrad(bf, s, 1) # q value and derivative w.r.t. that action in state s. 
```

Here is an example using the `FourierBasis`. 
```julia

dorder = 2  # order of the basis for all coupled terms. The number of features grow exponentially with this parameter. 
iorder = 4  # order of the basis for each individual state feature. The number of features grows linearly with this parameter. 
full = true  # if true it computes both sine and cosine of the features, otherwise only cosine will be computed
fb = FourierBasis(state_dims, dorder, iorder, full)  # assumes features are normalized to [0,1]
fbuff = FourierBasisBuffer(fb)  # creates buffer to avoid allocations
num_features = length(fb)  # gets the total number of features output by the basis
basisf = BufferedFunction(fb, fbuff)
ϕ = Chain(nf, basisf)

m = LinearModel(ϕ, num_features, num_actions=num_actions)
buff = LinearBuffer(m)
bf = BufferedFunction(m, buff)

bf([1.1, 0.0], 1)  # q value for first action at the give state features
v, g = value_withgrad(bf, [1.1, 0.0], 1)  # q value with partial derivative with respect to the model weights

```