
include("fourierbasis.jl")
include("tilecoding.jl")

struct ConcatentateBasis{TB} <: Any
    ϕs::TB
end

struct ConcatentateBasisBuffer{TO} <: Any
    out::TO

    function ConcatentateBasisBuffer(ϕ::ConcatentateBasis)
        n = sum(length(ϕ.ϕs))
        out = zeros(n)
        return new{typeof(out)}(out)
    end
end 

function (ϕ::ConcatentateBasis)(x)
    n = sum(length(ϕ.ϕs))
    out = zeros(n)
    tot = 1
    for i in 1:length(ϕ.ϕs)
        ni = length(ϕ.ϕs[i])
        out[tot:tot+ni-1] .= ϕ.ϕs[i](x)
        tot += ni
    end
    return out
end

function (ϕ::ConcatentateBasis)(buff::ConcatentateBasisBuffer, x)
    out = buff.out
    tot = 1
    for i in 1:length(ϕ.ϕs)
        ni = length(ϕ.ϕs[i])
        out[tot:tot+ni-1] .= ϕ.ϕs[i](buff.buffs[i], x)
        tot += ni
    end
    return out
end

function (ϕ::ConcatentateBasis)(buff, x)
    out = buff
    tot = 1
    for i in 1:length(ϕ.ϕs)
        y = ϕ.ϕs[i](x)
        ni = length(y)
        out[tot:tot+ni-1] .= y
        tot += ni
    end
    return out
end
