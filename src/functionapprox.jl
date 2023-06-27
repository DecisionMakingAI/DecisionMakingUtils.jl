

function value_withgrad(f::BufferedFunction, s)
    return value_withgrad(f.f, f.buff, s)
end

function value_withgrad(f::BufferedFunction, s, a)
    return value_withgrad(f.f, f.buff, s, a)
end


struct LinearBuffer{TO, TW,Tϕ}
    output::TO
    grad::TW
end

struct LinearModel{T,TW,Tϕ} 
    w::TW
    ϕ::Tϕ

    function LinearModel(w::TW, ϕ::Tϕ) where {TW,Tϕ}
        T = eltype(w)
        new{T,TW,Tϕ}(w, ϕ)
    end
end

function (m::LinearModel)(s)
    return m.w' * m.ϕ(s)
end

function (m::LinearModel)(buff::LinearBuffer, s)
    mul!(buff.output, m.w', m.ϕ(s))
    return buff.output
end

function (m::LinearModel)(s, a)
    w = @view m.w[..,a]
    return w' * m.ϕ(s)
end

function (m::LinearModel{T})(buff::LinearBuffer, s, a) where {T}
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

function lineargrad!(grad, w::AbstractMatrix{T}, x, a) where {T}
    @. grad[:, a] = x
end

function lineargrad!(grad, w::AbstractArray{T,3}, x, a) where {T}
    fill!(grad, zero(T))
    @. grad[.., a] = x
end

function value_withgrad(m::LinearModel, s)
    feats = m.ϕ(s)
    grad = zero(m.w)
    lineargrad!(grad, m.w, feats)
    return w' * feats, grad
end

function value_withgrad(m::LinearModel, s, a)
    feats = m.ϕ(s)
    grad = zero(m.w)
    lineargrad!(grad, m.w, feats, a)
    return w' * feats, grad
end

function value_withgrad(m::LinearModel{T}, buff::LinearBuffer, s) where {T}
    feats = m.ϕ(s)
    grad = buff.grad
    lineargrad!(grad, m.w, feats)
    return w' * feats, grad
end

function value_withgrad(m::LinearModel{T}, buff::LinearBuffer, s, a) where {T}
    feats = m.ϕ(s)
    grad = buff.grad
    lineargrad!(grad, m.w, feats, a)
    return w' * feats, grad
end

# W (num_outputs, num_tiles, num_tilings, num_actions)
struct TileCodingModel{T,TW,TC}
    w::TW
    ϕ::TC

    function TileCodingModel(w::TW, ϕ::TC) where {TW,TC}
        T = eltype(w)
        new{T,typeof(w), TC}(w, ϕ)
    end

    function TileCodingModel(ϕ::TC, num_outputs, num_tiles, num_tilings, num_actions=1) where {TW,TC}
        T = Float64
        w = zeros(T, (num_outputs, num_tiles, num_tilings, num_actions))
        w = dropdims(w, dims=(1,4))
        new{T,typeof(w),TC}(w, ϕ)
    end
end

function output_at_tile(w::AbstractArray{T,2}, idxs)  where {T}
    v = zero(T)
    for i in 1:length(idxs)
        v += w[idxs[i], i]
    end
    return v
end

function output_at_tile!(out::AbstractVector{T}, w::AbstractArray{T,3}, idxs)  where {T}
    fill!(out, zero(T))
    for i in 1:length(idxs)
        @. out += w[:, idxs[i], i]
    end
    return out
end

function output_at_tile!(out::AbstractMatrix{T}, w::AbstractArray{T,4}, idxs)  where {T}
    fill!(out, zero(T))
    numA = size(out, 2)
    for a in 1:numA
        for i in 1:length(idxs)
            @. out[:, a] += w[:, idxs[i], i, a]
        end 
    end
    return out
end

function output_at_tile(w::AbstractArray{T,3}, idxs, a)  where {T}
    v = zero(T)
    for i in 1:length(idxs)
        v += w[idxs[i], i, a]
    end
    return v
end

function output_at_tile!(out::AbstractMatrix{T}, w::AbstractArray{T,4}, idxs, a)  where {T}
    fill!(out, zero(T))
    for i in 1:length(idxs)
        @. out[:, a] += w[:, idxs[i], i, a]
    end 
    return out
end



function (m::TileCodingModel)(buff::LinearBuffer, s)
    idxs = m.ϕ(s)
    v = sum(i->getindex(m.w, (idxs[i], i)), idxs)
    return v
end

function (m::TileCodingModel)(s, a)
    idxs = m.ϕ(s)
    w = @view m.w[..,a]
    v = sum(i->getindex(w, (idxs[i], i)), idxs)
    return v
end

function value_withgrad(m::TileCodingModel, s)
    idxs = m.ϕ(s)
    v = sum(i->getindex(m.w, (idxs[i], i)), idxs)
    return v, idxs
end

function value_withgrad(m::TileCodingModel, s, a)
    idxs = m.ϕ(s)
    w = @view m.w[..,a]
    v = sum(i->getindex(w, (idxs[i], i)), idxs)
    return v, idxs
end

struct TabularModel{T,TW}
    w::TW

    function TabularModel(w::TW) where {TW}
        T = eltype(w)
        new{T,TW}(w)
    end
end

function (m::TabularModel)(s)
    return m.w[s]
end

function value_withgrad(m::TabularModel, s)
    return m.w[s], s
end

