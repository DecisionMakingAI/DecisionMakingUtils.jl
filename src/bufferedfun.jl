struct BufferedFunction{TF,TB} <: Any
    f::TF
    buff::TB

    function BufferedFunction(f, buff)
        new{typeof(f), typeof(buff)}(f, buff)
    end
end

function (f::BufferedFunction)(x)
    return f.f(f.buff, x)
end
