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

X = env.X  
state_dims = size(X, 1)
num_tiles = 5
num_tilings = 4
num_actions = length(env.A)

nrm = ZeroOneNormalization(X)
nbuff = zeros(num_observations)
nf = BufferedFunction(nrm, nbuff)
tc = TileCodingBasis(state_dims, num_tiles, num_tilings=num_tilings, tiling_type=:wrap)
ϕ = Chain(nf, tc)

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
