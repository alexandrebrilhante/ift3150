using DataFrames, ForwardDiff, Optim

df = readtable("data/aus/model_australia.txt", separator = ' ', header = false)

# β => γ, x0 => β

function cg(A::Matrix, b::Vector, β0::Vector)
    δ = 1e-6
    n = length(x0)
    β = β0
    g = b+A*β
    d = -g
    k = 0
    δ2 = δ*δ
    while dot(g, g) > δ2
        Ad = A*d
        normd = dot(d, Ad)
        α = -dot(d, g)/normd
        β += α*d
        g = b+A*β
        γ = dot(g, Ad)/normd
        d = -g+γ*d
        k += 1
    end
    normd = dot(d, A*d)
    α = -dot(d, g)/normd
    β += α*d
    return β, k
end

function f(β::Vector)
    i = 1
    m = 0
    while i < 1470
        c = 0
        d = 0
        data = convert(Array, df[i+3:i+5, :])
        choice = convert(Array, df[i+6:i+6, :])
        id = find(choice .== 1)
        alt = find(choice .== 0)
        c = exp(dot(vec(data[:, id]), β))
        for j in 1:length(alt)
            d += exp(dot(vec(data[:, alt[j]]), β))
        end
        m += log(c/(c+d))
        i += 7
    end
    return m/210
end

β, iter, k = cg_quadratic_tol(A, b, [0, 0, 0])
