using ForwardDiff
using Optim

immutable BasicTrustRegion{T <: Real}
    η1::T
    η2::T
    γ1::T
    γ2::T
end

function BTRDefaults()
    return BasicTrustRegion(0.01, 0.9, 0.5, 0.5)
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
    # If the iteration is successful, update the iterate
    if state.ρ >= b.η1
        true
    end
    false
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
    -τ*g*Δ/normg
end

function btr(f::Function, g!::Function, H!::Function,
        β0::Vector, tol::Float64 = 1e-8, verbose::Bool = false)
    b = BTRDefaults()
    state = BTRState()
    state.iter = 0
    state.Δ = 1.0
    state.β = β0
    n = length(β0)
    tol2 = tol*tol
    state.g = zeros(n)
    H = zeros(n, n)
    fβ = f(β0)
    g!(β0, state.g)
    H!(β0, H)
    nmax = 100000

    function model(s::Vector, g::Vector, H::Matrix)
        dot(s, g)+0.5*dot(s, H*s)
    end

    while dot(state.g, state.g) > tol2 && state.iter < nmax
        # Compute the step by approximately minimize the model
        state.step = CauchyStep(state.g, H, state.Δ)
        state.βcand = state.β+state.step
        # Compute the actual reduction over the predicted reduction
        fcand = f(state.βcand)
        state.ρ = (fcand-fβ)/(model(state.step, state.g, H))
        if acceptCandidate!(state, b)
            state.β = copy(state.βcand)
            g!(state.β, state.g)
            H!(state.β, H)
            fβ = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    state
end

f(β::Vector) = -10*β[1]^2+10*β[2]^2+4*sin(β[1]*β[2])-2*β[1]+β[1]^4

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

state = btr(f, g!, H!, [0,0], 1e-8)

state = btr(f, g!, H!, [0,0], 1e-7)
