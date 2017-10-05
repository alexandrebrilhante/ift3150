using DataFrames, ForwardDiff, Optim

df = readtable("data/aus/model_australia.txt", separator = ' ', header = false)

function newton(f::Function, g::Function, h::Function, β0::Vector)
    δ::Float64 = 1e-6
    nmax::Int64 = 1000
    k = 1
    β = β0
    n = length(β)
    δ2 = δ*δ
    H = eye(n)
    dfβ = ones(n)
    g(β, dfβ)
    while (dot(dfβ, dfβ) > δ2 && k <= nmax)
        k += 1
        g(β, dfβ)
        h(β, H)
        β -= H\dfβ
    end
    β, k
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
    while i <= 210
        c = 0
        d = 0
        data = convert(Array, df[i*7-3:i*7-1, :])
        choice = convert(Array, df[i*7:i*7, :])
        id = find(choice .== 1)
        alt = find(choice .== 0)
        c = exp(dot(vec(data[:, id]), β))
        for j in 1:length(alt)
            d += exp(dot(vec(data[:, alt[j]]), β))
        end
        m += log(c/(c+d))
        i += 1
    end
    return m/210
end

println(newton(f, g!, H!, [0, 0, 0]))

# Solution: ([0.0283255, -0.0257532, -0.00362244], 6)
