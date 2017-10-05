using DataFrames, ForwardDiff, Optim

df = readtable("data/aus/model_australia.txt", separator = ' ', header = false)

immutable BasicTrustRegion{T <: Real}
    η1::T
    η2::T
    γ1::T
    γ2::T
end

function BTRDefaults()
    return BasicTrustRegion(0.5, 0.5, 0.5, 0.5)
end

type BTRState
    iter::Int
    β::Vector
    βcand::Vector
    g::Vector
    step::Vector
    Δ::Float64
    ρ::Float64

    function BTRState()
        return new()
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

function CauchyStep(g::Vector, H::Matrix, Δ::Float64)
    q = dot(g, H*g)
    normg = norm(g)
    if q <= 0
        τ = 1.0
    else
        τ = min((normg*normg*normg)/(q*Δ), 1.0)
    end
    return -τ*g*Δ/normg
end

function btr(f::Function, g!::Function, H!::Function, β0::Vector)
    δ::Float64 = 1e-6
    b = BTRDefaults()
    state = BTRState()
    state.iter = 0
    state.Δ = 1.0
    state.β = β0
    n = length(β0)
    δ2 = δ*δ
    state.g = zeros(n)
    H = zeros(n, n)
    fβ = f(β0)
    g!(β0, state.g)
    H!(β0, H)
    nmax = 1000

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end

    while (dot(state.g, state.g) > δ2 && state.iter <= nmax)
        state.step = CauchyStep(state.g, H, state.Δ)
        state.βcand = state.β+state.step
        fcand = f(state.βcand)
        state.ρ = (fcand-fβ)/(model(state.step, state.g, H))
        if (acceptCandidate!(state, b))
            state.β = copy(state.βcand)
            g!(state.β, state.g)
            H!(state.β, H)
            fβ = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    return state.β, state.iter
end

g = β -> ForwardDiff.gradient(f, β);
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
    return -m/210
end

println(btr(f, g!, H!, [0, 0, 0]))

# Solution: ([0.0283255, -0.0257532, -0.00362244], 39)

