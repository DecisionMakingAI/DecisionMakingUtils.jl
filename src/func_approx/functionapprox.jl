struct LinearBuffer{TO,TG}
    output::TO
    grad::TG
end

include("linearmodel.jl")
include("tilecodingmodel.jl")


function (f::BufferedFunction)(s, a::Int)
    return f.f(f.buff, s, a)
end

function value_withgrad(f::BufferedFunction, s)
    return value_withgrad(f.buff, f.f, s)
end

function value_withgrad(f::BufferedFunction, s, a::Int)
    return value_withgrad(f.buff, f.f, s, a)
end




# struct TabularModel{T,TO,TA}
#     w::TW

#     function TabularModel(w::AbstractArray{T,3}) where {T}
#         no, ns, na = size(w)
#         TO = no == 1
#         TA = na == 1
#         new{T,TO,TA}(w)
#     end

#     function TabularModel(::Type{T}, num_states; num_outputs=1, num_actions=1) where {T}
#         w = zeros(T, (num_states, num_outputs, num_actions))
#         TO = num_outputs == 1
#         TA = num_actions == 1
#         new{T,TO,TA}(w)
#     end

#     function TabularModel(num_states; num_outputs=1, num_actions=1)
#         TabularModel(Float64, num_states, num_outputs=num_outputs, num_actions=num_actions)
#     end
# end

# function (m::TabularModel{T,true,true})(s) where {T}
#     return m.w[1,s,1]
# end

# function (m::TabularModel{T,true,true})(s, a::Int) where {T}
#     na = size(m.w, 3)
#     @assert (a ≥ 1) && (a ≤ na) "Invalid action index: $a ∉ [1, $na]"
#     return m.w[1,s,a]
# end

