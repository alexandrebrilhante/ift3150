using BenchmarkTools, Compat, DataFrames, Distributions, ForwardDiff

# Basic trust region with conjugate gradient.

df = readtable("../data/model_australia.txt", separator = ' ', header = false)

names!(df, [Symbol("C$i") for i in 1:4])

mixed_logit = DataFrame(P = 1.0:210.0)

names!(mixed_logit, [Symbol("Score")])

rand_contdist(Dist::Distribution) = quantile(Dist, rand())

function simulate()
    for i = 1:210
        mixed_logit[i, 1] = rand_contdist(Uniform())
    end
end

simulate()

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

function f(θ::Vector, model::Float64 = 0.0, n::Int64 = 210)
    i = 1
    while i <= n
        probability = individual(θ, i)
        model += log(probability(θ))
        i += 1
    end
    return -model/n
end

immutable BasicTrustRegion{T<:Real}
    η1::T
    η2::T
    γ1::T
    γ2::T
end

function BTRDefaults()
    return BasicTrustRegion(0.01, 0.9, 0.5, 0.5)
end

type BTRState
    iter::Int64
    x::Vector
    xcand::Vector
    g::Vector
    step::Vector
    Δ::Float64
    ρ::Float64
    tol::Float64

    function BTRState()
        state = new()
        state.tol = 1e-6
        return state
    end
end

function acceptCandidate!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η1
        return true
    else
        return false
    end
end

function updateRadius!(state::BTRState, b::BasicTrustRegion)
    if state.ρ >= b.η2
        stepnorm = norm(state.step)
        state.Δ = min(1e20, max(4*stepnorm, state.Δ))
    elseif state.ρ >= b.η1
        state.Δ *= b.γ2
    else
        state.Δ *= b.γ1
    end
end

function ConjugateGradient(A::Matrix, b::Vector, x0::Vector,
        tol::Float64 = 1e-6)
    n = length(x0)
    x = x0
    g = b+A*x
    d = -g
    k = 0
    tol2 = tol*tol
    while dot(g, g) > tol2
        Ad = A*d
        normd = dot(d, Ad)
        α = -dot(d, g)/normd
        x += α*d
        g = b+A*x
        γ = dot(g, Ad)/normd
        d = -g+γ*d
        k += 1
    end
    normd = dot(d, A*d)
    α = -dot(d, g)/normd
    x += α*d
    return x
end

function btr(f::Function, g!::Function, H!::Function, Step::Function,
        x0::Vector, state::BTRState = BTRState(), ApproxH::Bool = false)
    b = BTRDefaults()
    state.iter = 0
    state.x = x0
    n = length(x0)
    tol2 = state.tol*state.tol
    state.g = zeros(n)
    H = eye(n, n)
    fx = f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    if ApproxH
        y = zeros(n)
        gcand = zeros(n)
    else
        H!(x0, H)
    end
    nmax = 100

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end

    while dot(state.g, state.g) > tol2 && state.iter < nmax
        state.step = Step(H, state.g, x0)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-fx)/(model(state.step, state.g, H))
        if ApproxH
            g!(state.xcand, gcand)
            y = gcand-state.g
            H = H!(H, y, state.step)
        end
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            if ApproxH == false
                g!(state.x, state.g)
                H!(state.x, H)
            else
                state.g = copy(gcand)
            end
            fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    return state.x, state.iter
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

println(btr(f, g!, H!, ConjugateGradient, zeros(7), BTRState()))

#println(btr(f, g!, BHHH!, ConjugateGradient, zeros(7), BTRState()))

println(btr(f, g!, BFGS!, ConjugateGradient, zeros(7), BTRState(), true))

println(btr(f, g!, SR1!, ConjugateGradient, zeros(7), BTRState(), true))
