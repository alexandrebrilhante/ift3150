using DataFrames, ForwardDiff, Optim

df = readtable("data/aus/model_australia.txt", separator = ' ', header = false)

function newton(f::Function, g::Function, h::Function, β0::Vector)
    δ::Float64 = 1e-8
    nmax::Int64 = 100000
    k = 1
    β = β0
    n = length(β)
    δ2 = δ*δ
    H = eye(n)
    dfβ = ones(n)
    g(β, dfβ)
    while dot(dfβ, dfβ) > δ2 && k <= nmax
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
    while i < 1470
        data = convert(Array, df[i+3:i+5, 1:4])
        choice = convert(Array, df[i+6:i+6, 1:4])
        id = find(choice .== 1)
        alt = find(choice .== 0)
        n = exp(β[1]*data[1, id][1]+β[2]*data[2, id][1]+β[3]*data[3, id][1])
        d1 = exp(β[1]*data[1, alt][1]+β[2]*data[2, alt][1]+β[3]*data[3, alt][1])
        d2 = exp(β[1]*data[1, alt][2]+β[2]*data[2, alt][2]+β[3]*data[3, alt][2])
        d3 = exp(β[1]*data[1, alt][3]+β[2]*data[2, alt][3]+β[3]*data[3, alt][3])
        m += log(n/(n+d1+d2+d3))
        i += 7
    end
    m/210
end

println(newton(f, g!, H!, [0, 0, 0]))
