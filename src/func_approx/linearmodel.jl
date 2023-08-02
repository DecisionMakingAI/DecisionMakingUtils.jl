"""
    LinearModel([::Type,] ϕ, num_features; num_outputs=1, num_actions=1)

This creates a linear function that uses the basis function ϕ. It can have multiple outputs (num_outputs >1) and optionally make action dependent predictions (num_actions > 1)

See also: [`TileCodingModel`](@ref), [`TabularModel`](@ref)

# Examples
```jldoctest
julia> ϕ = x->[x, x^2];

julia> m = LinearModel(ϕ, 2);

julia> vec(params(m)) .= 1:length(params(m));

julia> m(2.0) # 1 * 2 + 2 * 2^2
10.0

julia> m = LinearModel(ϕ, 2, num_outputs=2,num_actions=3);

julia> vec(params(m)) .= 1:length(params(m));

julia> m(2.0)
2×3 Matrix{Float64}:
 14.0  38.0  62.0
 20.0  44.0  68.0

julia> m(2.0, 3)  # 3rd action prediction
2-element Vector{Float64}:
 62.0
 68.0

julia> v,g = value_withgrad(m, 2.0, 1);

julia> v
2-element Vector{Float64}:
 14.0
 20.0

julia> g
2×2×3 Array{Float64, 3}:
[:, :, 1] =
 2.0  4.0
 2.0  4.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 3] =
 0.0  0.0
 0.0  0.0
 
```
"""
struct LinearModel{T,TO,TA,Tϕ} 
    w::Array{T,3}
    ϕ::Tϕ

    function LinearModel(::Type{T}, ϕ::Tϕ, num_features::Int; num_outputs::Int=1, num_actions::Int=1) where {T,Tϕ}
        w = zeros(T, (num_outputs, num_features, num_actions))
        TO = num_outputs == 1
        TA = num_actions == 1
        return new{T,TO,TA,Tϕ}(w, ϕ)
    end

    function LinearModel(ϕ::Tϕ, num_features::Int; num_outputs::Int=1, num_actions::Int=1) where {Tϕ}
        return LinearModel(Float64, ϕ, num_features, num_outputs=num_outputs, num_actions=num_actions)
    end
end

function params(m::LinearModel)
    return m.w
end

function LinearBuffer(m::LinearModel{T}) where {T}
    no, nf, na = size(m.w)
    output = zeros(T, no, na)
    grad = zero(m.w)
    TO = typeof(output)
    TG = typeof(grad)
    return LinearBuffer{TO,TG}(output, grad)
end

function value(buff, m::LinearModel{T,true,true}, feats) where {T}
    wi = @view m.w[1,:,1]
    v = dot(wi, feats)
    return v
end

function value(buff, m::LinearModel{T,true,false}, feats) where {T}
    w = @view m.w[1,:,:]
    mul!(buff, w', feats)
    return buff
end

function value(buff, m::LinearModel{T,false,true}, feats) where {T}
    w = @view m.w[:,:,1]
    mul!(buff, w, feats)
    return buff
end

function value(buff, m::LinearModel{T,false,false}, feats) where {T}
    no, nf, na = size(m.w)
    for a in 1:na
        b = @view buff[:,a]
        w = @view m.w[:,:,a]
        mul!(b, w, feats)
    end
    return buff
end

function value(buff, m::LinearModel{T,true,true}, feats, a::Int) where {T}
    @assert a == 1 "Not a valid action for single action model, $a ≠ 1"
    wi = @view m.w[1,:,1]
    v = dot(wi, feats)
    return v
end

function value(buff, m::LinearModel{T,true,false}, feats, a::Int) where {T}
    na = size(m.w, 3)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    w = @view m.w[..,a]
    v = dot(w, feats)
    return v
end

function value(buff, m::LinearModel{T,false,true}, feats, a::Int) where {T}
    @assert a == 1 "Not a valid action for single action model, $a ≠ 1"
    w = @view m.w[:,:,1]
    mul!(buff, w, feats)
    return buff
end

function value(buff, m::LinearModel{T,false,false}, feats, a::Int) where {T}
    na = size(m.w, 3)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    w = @view m.w[:,:,a]
    mul!(buff, w, feats)
    return buff
end

function (m::LinearModel)(buff, s)
    feats = m.ϕ(s)
    v = value(buff.output, m, feats)
    return v
end

function (m::LinearModel{T,true,false})(buff::LinearBuffer, s) where {T}
    feats = m.ϕ(s)
    output = @view buff.output[1,:]
    v = value(output, m, feats)
    return v
end

function (m::LinearModel{T,false,true})(buff::LinearBuffer, s) where {T}
    feats = m.ϕ(s)
    output = @view buff.output[:,1]
    v = value(output, m, feats)
    return v
end

function (m::LinearModel)(buff, s, a::Int)
    feats = m.ϕ(s)
    na = size(m.w,3)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    out = @view buff.output[:, a]
    v = value(out, m, feats, a)
    return v
end

function make_outputbuff(m::LinearModel{T,true,true}) where {T}
    return nothing
end

function make_outputbuff(m::LinearModel{T,true,true}, a::Int) where {T}
    return nothing
end

function make_outputbuff(m::LinearModel{T,true,false}, a::Int) where {T}
    return nothing
end

function make_outputbuff(m::LinearModel{T}) where {T}
    no, nf, na = size(m.w)
    return zeros(T, (no, na))
end

function make_outputbuff(m::LinearModel{T,true,false}) where {T}
    no, nf, na = size(m.w)
    return zeros(T, na)
end

function make_outputbuff(m::LinearModel{T,false,true}) where {T}
    no, nf, na = size(m.w)
    return zeros(T, no)
end

function make_outputbuff(m::LinearModel{T}, a::Int) where {T}
    no, nf, na = size(m.w)
    return zeros(T, no)
end

function (m::LinearModel)(s)
    feats = m.ϕ(s)
    buff = make_outputbuff(m)
    v = value(buff, m, feats)
    return v
end

function (m::LinearModel)(s, a::Int)
    feats = m.ϕ(s)
    buff = make_outputbuff(m, a)
    v = value(buff, m, feats, a)
    return v
end

function lineargrad!(grad, feats)
    T = eltype(grad)
    fill!(grad, zero(T))
    @. grad = feats'
    return nothing
end

function lineargrad!(grad, feats, a::Int) 
    T = eltype(grad)
    fill!(grad, zero(T))
    @. grad[:,:,a] = feats'
    return nothing
end

function value_withgrad(m::LinearModel, s)
    feats = m.ϕ(s)
    out = make_outputbuff(m)
    v = value(out, m, feats)
    grad = zero(m.w)
    lineargrad!(grad, feats)
    return v, grad
end

function value_withgrad(m::LinearModel, s, policy::TF) where {TF}
    buff = LinearBuffer(m)
    return value_withgrad(buff, m, s, policy)
end

function value_withgrad(m::LinearModel, s, a::Int)
    feats = m.ϕ(s)
    out = make_outputbuff(m, a)
    v = value(out, m, feats, a)
    grad = zero(m.w)
    lineargrad!(grad, feats, a)
    return v, grad
end

function value_withgrad(buff, m::LinearModel, s) 
    feats = m.ϕ(s)
    v = value(buff.output, m, feats)
    grad = buff.grad
    lineargrad!(grad, feats)
    return v, grad
end

function value_withgrad(buff, m::LinearModel{T,true,false}, s) where {T}
    feats = m.ϕ(s)
    output = @view buff.output[1,:]
    v = value(output, m, feats)
    grad = buff.grad
    lineargrad!(grad, feats)
    return v, grad
end

function value_withgrad(buff, m::LinearModel{T,true,false}, s, policy::TF) where {T, TF<:Function}
    feats = m.ϕ(s)
    output = @view buff.output[1,:]
    v = value(output, m, feats)
    a = policy(v)
    na = size(m.w,3)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    grad = buff.grad
    lineargrad!(grad, feats, a)
    return a, v[a], grad
end

function value_withgrad(buff, m::LinearModel{T,false,true}, s) where {T}
    feats = m.ϕ(s)
    output = @view buff.output[:,1]
    v = value(output, m, feats)
    grad = buff.grad
    lineargrad!(grad, feats)
    return v, grad
end

function value_withgrad(buff, m::LinearModel{T,TB,true}, s, policy::TF) where {T, TB,TF<:Function}
    feats = m.ϕ(s)
    output = @view buff.output[:,1]
    v = value(output, m, feats)
    a = policy(v)
    @assert a == 1 "Not a valid action: $a ∉ [1, 1]"
    grad = buff.grad
    lineargrad!(grad, feats, a)
    return a, v, grad
end

function value_withgrad(buff, m::LinearModel, s, a::Int) 
    feats = m.ϕ(s)
    na = size(m.w,3)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    out = @view buff.output[:, a]
    v = value(out, m, feats, a)
    grad = buff.grad
    lineargrad!(grad, feats, a)
    return v, grad
end

function value_withgrad(buff, m::LinearModel, s, policy::TF) where {TF<:Function} 
    feats = m.ϕ(s)
    na = size(m.w,3)
    v = value(buff.output, m, feats)
    a = policy(v)
    @assert a ≥ 1 && a ≤ na "Not a valid action, $a ∉ [1, $na]"
    grad = buff.grad
    lineargrad!(grad, feats, a)
    return a, v[..,a], grad
end