
using Optim

immutable BasicTrustRegion{T <: Real}
    η1:: T
    η2:: T
    γ1:: T
    γ2:: T
end

function BTRDefaults()
    return BasicTrustRegion(0.01,0.9,0.5,0.5)
end

type BTRState
    iter::Int
    x::Vector
    xcand::Vector
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
    if (state.ρ >= b.η1)
        return true
    else
        return false
    end
end

function updateRadius!(state::BTRState, b::BasicTrustRegion)
    if (state.ρ >= b.η2)
        stepnorm = norm(state.step)
        state.Δ = min(1e20,max(4*stepnorm,state.Δ))
    elseif (state.ρ >= b.η1)
        state.Δ *= b.γ2
    else
        state.Δ *= b.γ1
    end
end

function CauchyStep(g::Vector,H::Matrix,Δ::Float64)
    q = dot(g,H*g)
    normg = norm(g)
    if (q <= 0)
        τ = 1.0
    else
        τ = min((normg*normg*normg)/(q*Δ),1.0)
    end
    return -τ*g*Δ/normg
end

function btr(f::Function, g!::Function, H!::Function,
    x0::Vector, tol::Float64 = 1e-8, verbose::Bool = false)

    b = BTRDefaults()
    state = BTRState()
    state.iter = 0
    state.Δ = 1.0
    state.x = x0
    n=length(x0)

    tol2 = tol*tol

    state.g = zeros(n)
    H = zeros(n,n)

    fx = f(x0)
    g!(x0, state.g)
    H!(x0, H)

    nmax = 100000

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end

    while (dot(state.g,state.g) > tol2 && state.iter < nmax)
        # Compute the step by approximately minimize the model
        state.step = CauchyStep(state.g, H, state.Δ)
        state.xcand = state.x+state.step

        # Compute the actual reduction over the predicted reduction
        fcand = f(state.xcand)
        state.ρ = (fcand-fx)/(model(state.step, state.g, H))

        if (acceptCandidate!(state, b))
            state.x = copy(state.xcand)
            g!(state.x, state.g)
            H!(state.x, H)
            fx = fcand
        end

        updateRadius!(state, b)
        state.iter += 1
    end

    return state
end

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function rosenbrock_gradient!(x::Vector, storage::Vector)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

function rosenbrock_hessian!(x::Vector, storage::Matrix)
    storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    storage[1, 2] = -400.0 * x[1]
    storage[2, 1] = -400.0 * x[1]
    storage[2, 2] = 200.0
end

state = btr(rosenbrock, rosenbrock_gradient!, rosenbrock_hessian!, [0,0])

state

using ForwardDiff

f(x::Vector) = -10*x[1]^2+10*x[2]^2+4*sin(x[1]*x[2])-2*x[1]+x[1]^4

g = x -> ForwardDiff.gradient(f, x);
H = x -> ForwardDiff.hessian(f, x)

function g!(x::Vector, storage::Vector)
    s = g(x)
    storage[1:length(s)] = s[1:length(s)]
end

function H!(x::Vector, storage::Matrix)
    s = H(x)
    n, m = size(s)
    storage[1:n,1:m] = s[1:length(s)]
end

state = btr(f, g!, H!, [0,0],1e-8)

state = btr(f, g!, H!, [0,0],1e-7)
