using DataFrames, ForwardDiff, Optim

df = readtable("data/aus/model_australia.txt", separator = ' ', header = false)

function newton(f::Function, g::Function, h:: Function,
        β0::Vector, δ::Float64 = 1e-8, nmax::Int64 = 100000)
    k = 1
    β = β0
    n = length(β)
    δ2 = δ*δ
    H = eye(n)
    dfβ = ones(n)
    while dot(dfβ, dfβ) > δ2 #&& k <= nmax
        k += 1
        g(β, dfβ)
        h(β, H)
        β -= H\dfβ
    end
    β
end

g = β -> ForwardDiff.gradient(f, β)
H = β -> ForwardDiff.hessian(f, β)

function g!(β::Vector, storage::Vector)
    s = g(β)
    storage[1:length(s)] = s[1:length(s)]
end

function H!(β::Vector, storage::Matrix)
    s = H(β)
    n, m = size(s)
    storage[1:n, 1:m] = s[1:length(s)]
end

function f(β::Vector)
    i = 1
    m = 0
    while i < 1465
        data = convert(Array, df[i:i+5, 1:4])
        choice = convert(Array, df[i+6:i+6, 1:4])
        id = find(choice .== 1)
        alt = find(choice .== 0)
        n = exp(β[1]*data[4, id][1]+β[2]*data[5, id][1]+β[3]*data[6, id][1])
        d = (n+
            exp(β[1]*data[4, alt[1]]+β[2]*data[4, alt[2]]+β[3]*data[4, alt[3]])+
            exp(β[1]*data[5, alt[1]]+β[2]*data[5, alt[2]]+β[3]*data[5, alt[3]])+
            exp(β[1]*data[6, alt[1]]+β[2]*data[6, alt[2]]+β[3]*data[6, alt[3]]))
        m += log(n/d)
        i += 7
    end
    m/210
end

println(newton(f, g!, H!, [0, 0, 0]))
