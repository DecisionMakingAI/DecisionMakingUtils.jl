struct LinearModel{T,TW,Tϕ} 
    w::TW
    ϕ::Tϕ

    function LinearModel(w::TW, ϕ::Tϕ) where {TW,Tϕ}
        T = eltype(w)
        new{T,TW,Tϕ}(w, ϕ)
    end
end

function params(m::LinearModel)
    return m.w
end

function (m::LinearModel)(s)
    return m.w' * m.ϕ(s)
end

function (m::LinearModel)(buff, s)
    mul!(buff.output, m.w', m.ϕ(s))
    return buff.output
end

function (m::LinearModel)(s, a::Int)
    w = @view m.w[..,a]
    return w' * m.ϕ(s)
end

function (m::LinearModel{T})(buff, s, a::Int) where {T}
    w = @view m.w[..,a]
    mul!(buff.output, w', m.ϕ(s))
    return buff.output
end

function lineargrad!(grad, w::AbstractVector{T}, x) where {T}
    fill!(grad, zero(T))
    @. grad = x
end

function lineargrad!(grad, w::AbstractMatrix{T}, x) where {T}
    fill!(grad, zero(T))
    @. grad = x
end

function lineargrad!(grad, w::AbstractMatrix{T}, x, a::Int) where {T}
    @. grad[:, a] = x
end

function lineargrad!(grad, w::AbstractArray{T,3}, x, a::Int) where {T}
    fill!(grad, zero(T))
    @. grad[.., a] = x
end

function value_withgrad(m::LinearModel, s)
    feats = m.ϕ(s)
    grad = zero(m.w)
    lineargrad!(grad, m.w, feats)
    return w' * feats, grad
end

function value_withgrad(m::LinearModel, s, a::Int)
    feats = m.ϕ(s)
    grad = zero(m.w)
    lineargrad!(grad, m.w, feats, a)
    return w' * feats, grad
end

function value_withgrad(buff, m::LinearModel{T}, s) where {T}
    feats = m.ϕ(s)
    grad = buff.grad
    lineargrad!(grad, m.w, feats)
    return w' * feats, grad
end

function value_withgrad(buff, m::LinearModel{T}, s, a::Int) where {T}
    feats = m.ϕ(s)
    grad = buff.grad
    lineargrad!(grad, m.w, feats, a)
    return w' * feats, grad
end