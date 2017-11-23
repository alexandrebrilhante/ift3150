using BenchmarkTools, Compat, DataFrames, Distributions, ForwardDiff

df = readtable("../data/model_australia.txt", separator = ' ', header = false)

names!(df, [Symbol("x$i") for i in 1:4])

mixed_logit = DataFrame(P = 1.0:210.0)

names!(mixed_logit, [Symbol("Score")])

rand_contdist(Dist::Distribution) = quantile(Dist, rand())

function simulate()
    for i = 1:210
        mixed_logit[i, 1] = rand_contdist(Uniform())
    end
end

function individual(θ::Vector, i::Int64)
    data = convert(Array, df[i*7-6:i*7-1, :])
    choices = convert(Array, df[i*7:i*7, :])
    alternatives = find(choices .== 0)
    choice = find(choices .== 1)[1]

    function utility(β::Vector, i::Int64)
        return dot(vec(data[:, i]), β)
    end

    function construct(γ::Vector, θ::Vector, β::Float64 = 0.0)
        for k = 1:length(γ)
            β += θ[k*1]+θ[k*2]*γ[k]
        end
        return β
    end

    function probability(θ::Vector, t::Float64 = 0.0)
        γ = mixed_logit[i, 1]
        β = θ[1:4]
        push!(β, construct([γ], θ[5:6]))
        push!(β, θ[length(θ)])
        c = utility(β, choice)
        for alternative in alternatives
            t += exp(utility(β, alternative)-c)
        end
        return 1/(1+t)
    end

    return probability
end

simulate()

function f(θ::Vector, model::Float64 = 0.0, n::Int64 = 210)
    i = 1
    while i <= n
        probability = individual(θ, i)
        model += log(probability(θ))
        i += 1
    end
    return -model/n
end

function g(x::Vector, n::Int64 = 210)
    t = zeros(length(x))
    for i = 1:n
        probability = individual(x, i)
        t += (1/probability(x))*ForwardDiff.gradient(probability, x)
    end
    return -t/n
end

function g!(x::Vector, storage::Vector)
    s = g(x)
    storage[1:length(s)] = s[1:length(s)]
end

function H(x::Vector)
    return ForwardDiff.hessian(f, x)
end

function H!(x::Vector, storage::Matrix)
    s = H(x)
    n, m = size(s)
    storage[1:n, 1:m] = s[1:length(s)]
end

function BHHH(x::Vector, n::Int64 = 210)
    t = zeros(length(x), length(x))
    for i = 1:n
        probability = individual(x, i)
        g = ForwardDiff.gradient(probability, x)
        t += g*(g')
    end
    return t/n
end

function BHHH!(x::Vector, storage::Matrix)
    s = BHHH(x)
    n, m = size(s)
    storage[1:n, 1:m] = s[1:length(s)]
end

function BFGS(B::Matrix, y::Vector, s::Vector)
    Bs = B*s
    return B-(Bs*Bs')/dot(s, Bs)+(y*y')/dot(s, y)
end

function BFGS!(B::Matrix, y::Vector, s::Vector)
    n, m = size(B)
    B[1:n, 1:m] = BFGS(B, y, s)
end

function SR1(B::Matrix, y::Vector, s::Vector)
    Bs = B*s
    return B+((y-Bs)*(y-Bs)')/((y-Bs)'*s)
end

function SR1!(B::Matrix, y::Vector, s::Vector)
    n, m = size(B)
    B[1:n, 1:m] = SR1(B, y, s)
end

function newton(f::Function, g::Function, h::Function, x0::Vector,
        δ::Float64 = 1e-6, nmax::Int64 = 1000)
    k = 1
    x = x0
    n = length(x)
    δ *=δ
    H = eye(n)
    dfx = ones(n)
    g(x, dfx)
    while dot(dfx, dfx) > δ && k <= nmax
        k += 1
        g(x, dfx)
        h(x, H)
        x -= H\dfx
    end
    return x, k
end

println(newton(f, g!, H!, zeros(7)))
