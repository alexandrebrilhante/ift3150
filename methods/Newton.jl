using ForwardDiff

# data
# [n, ppg, ocg, ppe, oce, c]
# [1, 140, 130, 250, 350, 0]
# [2, 140, 130, 50, 45, 1]
# [3, 42, 13, 620, 450, 0]
# [4, 120, 250, 23, 98, 1]
# [5, 90, 20, 140, 130, 0]
# [6, 20, 50, 120, 190, 0]
# [7, 50, 150, 14, 13, 1]
# [8, 150, 50, 240, 230, 1]
# [9, 200, 150, 40, 30, 1]
# [0, 90, 80, 140, 120, 0]

function f(β::Vector)
    c1 = exp(β[1]*140+β[2]*130)/(exp(β[1]*140+β[2]*130)+exp(β[1]*250+β[2]*350))
    c2 = exp(β[1]*50+β[2]*45)/(exp(β[1]*50+β[2]*45)+exp(β[1]*140+β[2]*130))
    c3 = exp(β[1]*42+β[2]*13)/(exp(β[1]*42+β[2]*13)+exp(β[1]*620+β[2]*450))
    c4 = exp(β[1]*23+β[2]*98)/(exp(β[1]*23+β[2]*98)+exp(β[1]*120+β[2]*250))
    c5 = exp(β[1]*90+β[2]*20)/(exp(β[1]*90+β[2]*20)+exp(β[1]*140+β[2]*130))
    c6 = exp(β[1]*20+β[2]*50)/(exp(β[1]*20+β[2]*50)+exp(β[1]*120+β[2]*190))
    c7 = exp(β[1]*14+β[2]*13)/(exp(β[1]*14+β[2]*13)+exp(β[1]*50+β[2]*150))
    c8 = exp(β[1]*150+β[2]*50)/(exp(β[1]*150+β[2]*50)+exp(β[1]*240+β[2]*230))
    c9 = exp(β[1]*40+β[2]*30)/(exp(β[1]*40+β[2]*30)+exp(β[1]*200+β[2]*150))
    c0 = exp(β[1]*90+β[2]*80)/(exp(β[1]*90+β[2]*80)+exp(β[1]*140+β[2]*130))
    return (log(c1)+log(c2)+log(c3)+log(c4)+log(c5)+log(c6)+log(c7)+log(c8)+
    log(c9)+log(c0))/length(β)
end

g = x -> ForwardDiff.gradient(f, x)
H = x -> ForwardDiff.hessian(f, x)

function g!(x::Vector, storage::Vector)
    s = g(x)
    storage[1:length(s)] = s[1:length(s)]
end

function H!(x::Vector, storage::Matrix)
    s = H(x)
    n, m = size(s)
    storage[1:n, 1:m] = s[1:length(s)]
end

function newton(f::Function, g::Function, h:: Function,
        xstart::Vector, verbose::Bool = false,
        δ::Float64 = 1e-8, nmax::Int64 = 1000)
    k = 1
    x = xstart
    n = length(x)
    δ2 = δ*δ
    H = eye(n)
    dfx = ones(n)
    if verbose
        fx = f(x)
        println("$k. x = $x, f(x) = $fx")
    end
    g(x, dfx)
    while dot(dfx, dfx) > δ2 && k <= nmax
        k += 1
        g(x, dfx)
        h(x, H)
        x -= H\dfx
        if verbose
            fx = f(x)
            println("$k. x = $x, f(x) = $fx")
        end
    end
    x
end

f(newton(f, g!, H!, [0, 0]))
