# W (num_outputs, num_tiles, num_tilings, num_actions)
struct TileCodingModel{T,TO,TA,TC} 
    w::Array{T,4}
    ϕ::TC

    function TileCodingModel(::Type{T}, ϕ::TC; num_tiles, num_tilings, num_outputs=1, num_actions=1) where {T,TC}
        w = zeros(T, (num_outputs, num_tiles, num_tilings, num_actions))
        single_out = (num_outputs == 1)
        no_action = (num_actions == 1)
        new{T,single_out,no_action,TC}(w, ϕ)
    end

    function TileCodingModel(ϕ::TC; num_tiles, num_tilings, num_outputs=1, num_actions=1) where {TC}
        TileCodingModel(Float64, ϕ, num_tiles=num_tiles, num_tilings=num_tilings, num_outputs=num_outputs, num_actions=num_actions)
    end
end

function TabularModel(::Type{T}, num_states; num_outputs=1, num_actions=1) where {T}
    TileCodingModel(T, identity, num_tiles=num_states, num_tilings=1, num_outputs=num_outputs, num_actions=num_actions)
end

function TabularModel(num_states; num_outputs=1, num_actions=1)
    TabularModel(Float64, num_states, num_outputs=num_outputs, num_actions=num_actions)
end

function LinearBuffer(m::TileCodingModel{T}) where {T}
    no, na = size(m.w, 1), size(m.w, 4)
    output = zeros(T, no, na)
    grad = zero(m.w)
    TO = typeof(output)
    TG = typeof(grad)
    return LinearBuffer{TO,TG}(output, grad)
end

function params(m::TileCodingModel)
    return m.w
end

function output_at_tile(w::AbstractArray{T,4}, idxs)  where {T}
    v = zero(T)
    for i in 1:length(idxs)
        v += w[1, idxs[i], i, 1]
    end
    return v
end

function output_at_tile(w::AbstractArray{T,4}, idxs, a)  where {T}
    v = zero(T)
    for i in 1:length(idxs)
        v += w[1, idxs[i], i, a]
    end
    return v
end


function output_at_tile!(out::AbstractMatrix{T}, w::AbstractArray{T,4}, idxs)  where {T}
    fill!(out, zero(T))
    no, ns, nt, na = size(w)
    @assert (no, na) == size(out) "Output size mismatch"
    for i in 1:length(idxs)
        wi = @view w[:, idxs[i], i, :]
        @. out += wi
    end 
    return out
end

function output_at_tile!(out::AbstractVector{T}, w::AbstractArray{T,4}, idxs, a)  where {T}
    fill!(out, zero(T))
    @assert length(out) == size(w,1) "Output size mismatch"
    for i in 1:length(idxs)
        wi = @view w[:, idxs[i], i, a]
        @. out += wi
    end 
    return out
end

function output_at_tile!(out::AbstractVector{T}, w::AbstractArray{T,4}, idxs)  where {T}
    fill!(out, zero(T))
    no, _, _, na = size(w)
    @assert no == 1 "not single output model"
    @assert length(out) == na "Output size mismatch"
    for i in 1:length(idxs)
        wi = @view w[1, idxs[i], i, :]
        @. out += wi
    end 
    return out
end

function make_outputbuff(m::TileCodingModel{T,true,true}) where {T}
    return nothing
end

function make_outputbuff(m::TileCodingModel{T,true,true}, a) where {T}
    return nothing
end

function make_outputbuff(m::TileCodingModel{T,true,false}) where {T}
    return zeros(T, size(m.w, 4))
end

function make_outputbuff(m::TileCodingModel{T,true,false}, a) where {T}
    return nothing
end

function make_outputbuff(m::TileCodingModel{T,false,true}) where {T}
    return zeros(T, size(m.w, 1))
end

function make_outputbuff(m::TileCodingModel{T,false,true}, a) where {T}
    return zeros(T, size(m.w, 1))
end

function make_outputbuff(m::TileCodingModel{T,false,false}) where {T}
    no, _, _, na = size(m.w)
    return zeros(T, (no, na))
end

function make_outputbuff(m::TileCodingModel{T,false,false}, a) where {T}
    no = size(m.w, 1)
    return zeros(T, no)
end

function value(buff, m::TileCodingModel{T,true,true}, idxs) where {T}
    v = output_at_tile(m.w, idxs)
    return v
end

function value(buff, m::TileCodingModel{T,true,true}, idxs, a) where {T}
    @assert (a == 1) "Not a valid action: $a ∉ [1, 1]"
    v = output_at_tile(m.w, idxs)
    return v
end

function value(buff, m::TileCodingModel{T,true,false}, idxs) where {T}
    v = output_at_tile!(buff, m.w, idxs)
    return vec(v)
end

function value(buff, m::TileCodingModel{T,true,false}, idxs, a) where {T}
    na = size(m.w, 4)
    @assert ((a ≤ na) && (a ≥ 1)) "Not a valid action: $a ∉ [1, $na]"
    v = output_at_tile(m.w, idxs, a)
    return v
end

function value(buff, m::TileCodingModel{T,false,true}, idxs) where {T}
    v = output_at_tile!(buff, m.w, idxs,1)
    return vec(v)
end

function value(buff, m::TileCodingModel{T,false,true}, idxs, a) where {T}
    @assert (a == 1) "Not a valid action: $a ∉ [1, 1]"
    v = output_at_tile!(buff, m.w, idxs,1)
    return vec(v)
end

function value(buff, m::TileCodingModel{T,false,false}, idxs) where {T}
    v = output_at_tile!(buff, m.w, idxs)
    return v
end

function value(buff, m::TileCodingModel{T,false,false}, idxs, a) where {T}
    na = size(m.w, 4)
    @assert ((a ≤ na) && (a ≥ 1)) "Not a valid action: $a ∉ [1, $na]"
    v = output_at_tile!(buff, m.w, idxs, a)
    return v
end
# make sure buffer of prealloc works for each case

function (m::TileCodingModel)(buff, s)
    idxs = m.ϕ(s)
    return value(buff.output, m, idxs)
end

function (m::TileCodingModel)(buff, s, a::Int)
    idxs = m.ϕ(s)
    output = @view buff.output[:, a]
    return value(vec(output), m, idxs, a)
end

function (m::TileCodingModel)(s)
    idxs = m.ϕ(s)
    buff = make_outputbuff(m)
    return value(buff, m, idxs)
end

function (m::TileCodingModel)(s,a::Int) 
    idxs = m.ϕ(s)
    buff = make_outputbuff(m, a)
    return value(buff, m, idxs, a)
end

function grad_tile!(grad::AbstractArray{T,4}, idxs) where {T}
    for i in 1:length(idxs)
        @. grad[:, idxs[i], i, :] += 1
    end 
    return grad
end

function grad_tile!(grad::AbstractArray{T,4}, idxs, a) where {T}
    for i in 1:length(idxs)
        @. grad[:, idxs[i], i, a] += 1
    end 
    return grad
end

function value_withgrad(m::TileCodingModel, s)
    idxs = m.ϕ(s)
    buff = make_outputbuff(m)
    v = value(buff, m, idxs)
    grad = zero(m.w)
    grad_tile!(grad, idxs)
    return v, grad
end

function value_withgrad(m::TileCodingModel, s, a::Int)
    idxs = m.ϕ(s)
    buff = make_outputbuff(m, a)
    v = value(buff, m, idxs, a)
    grad = zero(m.w)
    grad_tile!(grad, idxs, a)
    return v, grad
end

function value_withgrad(buff, m::TileCodingModel, s)
    idxs = m.ϕ(s)
    v = value(buff.output, m, idxs)
    grad = buff.grad
    fill!(grad, zero(eltype(grad)))
    grad_tile!(grad, idxs)
    return v, grad
end

function value_withgrad(buff, m::TileCodingModel, s, a::Int)
    idxs = m.ϕ(s)
    na = size(m.w, 4)
    @assert ((a ≤ na) && (a ≥ 1)) "Not a valid action: $a ∉ [1, $na]"
    out = @view buff.output[:, a]
    v = value(out, m, idxs, a)
    grad = buff.grad
    fill!(grad, zero(eltype(grad)))
    grad_tile!(grad, idxs, a)
    return v, grad
end