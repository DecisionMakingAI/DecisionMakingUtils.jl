struct LinearBuffer{TO,TG}
    output::TO
    grad::TG
end

include("linearmodel.jl")
include("tilecodingmodel.jl")



function value_withgrad(f::BufferedFunction, s)
    return value_withgrad(f.buff, f.f, s)
end

function value_withgrad(f::BufferedFunction, s, a)
    return value_withgrad(f.buff, f.f, s, a)
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

