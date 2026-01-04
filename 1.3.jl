using CSV, DataFrames, Random, Printf, Statistics

function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

function normalize_vec(v::AbstractVector{<:Real}; eps=1e-12)
    w = v .+ eps
    return w ./ sum(w)
end

# Marsaglia–Tsang Gamma sampler
function gamma_rand(shape::Real, rng::AbstractRNG; scale::Real=1.0)
    if shape < 1
        u = rand(rng)
        return gamma_rand(shape + 1, rng; scale=scale) * u^(1/shape)
    end
    d = shape - 1/3
    c = 1 / sqrt(9d)
    while true
        x = randn(rng)
        v = 1 + c*x
        if v <= 0
            continue
        end
        v = v^3
        u = rand(rng)
        if u < 1 - 0.0331*(x^4)
            return scale * d * v
        end
        if log(u) < 0.5*x^2 + d*(1 - v + log(v))
            return scale * d * v
        end
    end
end

function dirichlet_sample(α::AbstractVector{<:Real}, rng::AbstractRNG)
    g = [gamma_rand(ai, rng) for ai in α]
    return g ./ sum(g)
end

# Core: EM for mixture of Markov chains
function em_mix_markov(X::Matrix{Int};
        K::Int=3, S::Int=3, max_iter::Int=200, tol::Float64=1e-6,
        seed::Int=42, eps::Float64=1e-12, verbose::Bool=false)

    N, T = size(X)
    rng = MersenneTwister(seed)

    # random Dirichlet initialisation
    θ = dirichlet_sample(ones(K), rng)        # mixture weights
    π = zeros(Float64, K, S)                  # initial distributions
    A = zeros(Float64, K, S, S)               # transitions

    for k in 1:K
        π[k, :] = dirichlet_sample(ones(S), rng)
        for s in 1:S
            A[k, s, :] = dirichlet_sample(ones(S), rng)
        end
    end

    loglik_hist = Float64[]

    for it in 1:max_iter
        # E-step
        logp = zeros(Float64, N, K)

        @inbounds for n in 1:N
            x1 = X[n, 1] + 1
            for k in 1:K
                lp = log(π[k, x1] + eps)
                for t in 2:T
                    s_prev = X[n, t-1] + 1
                    s_cur  = X[n, t]   + 1
                    lp += log(A[k, s_prev, s_cur] + eps)
                end
                logp[n, k] = lp
            end
        end

        log_joint = logp .+ reshape(log.(θ .+ eps), (1, K))
        γ = zeros(Float64, N, K)

        ll = 0.0
        @inbounds for n in 1:N
            lse = logsumexp(view(log_joint, n, :))
            ll += lse
            γ[n, :] = exp.(view(log_joint, n, :) .- lse)
        end
        push!(loglik_hist, ll)

        if verbose && (it == 1 || it % 10 == 0)
            @printf("iter %d, log-likelihood = %.6f\n", it, ll)
        end

        if it > 1 && abs(loglik_hist[end] - loglik_hist[end-1]) < tol
            if verbose
                @printf("Converged at iter %d, log-likelihood = %.6f\n", it, ll)
            end
            break
        end

        # M-step
        θ = normalize_vec(vec(mean(γ, dims=1)); eps=eps)

        # update π
        for k in 1:K
            counts = zeros(Float64, S)
            @inbounds for n in 1:N
                s1 = X[n, 1] + 1
                counts[s1] += γ[n, k]
            end
            π[k, :] = normalize_vec(counts; eps=eps)
        end

        # update A
        for k in 1:K
            num = zeros(Float64, S, S)
            den = zeros(Float64, S)

            @inbounds for n in 1:N
                w = γ[n, k]
                for t in 2:T
                    s_prev = X[n, t-1] + 1
                    s_cur  = X[n, t]   + 1
                    num[s_prev, s_cur] += w
                    den[s_prev] += w
                end
            end

            for s in 1:S
                if den[s] < eps
                    A[k, s, :] .= 1.0 / S
                else
                    A[k, s, :] = normalize_vec(num[s, :]; eps=eps)
                end
            end
        end
    end

    return θ, π, A, loglik_hist
end

function responsibilities(X::Matrix{Int}, θ, π, A; eps=1e-12)
    N, T = size(X)
    K = length(θ)
    logp = zeros(Float64, N, K)

    @inbounds for n in 1:N
        x1 = X[n, 1] + 1
        for k in 1:K
            lp = log(π[k, x1] + eps)
            for t in 2:T
                s_prev = X[n, t-1] + 1
                s_cur  = X[n, t]   + 1
                lp += log(A[k, s_prev, s_cur] + eps)
            end
            logp[n, k] = lp
        end
    end

    log_joint = logp .+ reshape(log.(θ .+ eps), (1, K))
    γ = zeros(Float64, N, K)

    @inbounds for n in 1:N
        lse = logsumexp(view(log_joint, n, :))
        γ[n, :] = exp.(view(log_joint, n, :) .- lse)
    end

    return γ
end

# Multi-start wrapper
function em_multistart(X::Matrix{Int};
        K::Int=3, S::Int=3, max_iter::Int=200, tol::Float64=1e-6,
        eps::Float64=1e-12, seeds=1:10, verbose_each::Bool=false)

    best_ll = -Inf
    best = nothing

    for sd in seeds
    θ_try, π_try, A_try, llhist_try = em_mix_markov(
        X; K=K, S=S, max_iter=max_iter, tol=tol, seed=sd, eps=eps, verbose=verbose_each
    )

    ll_try   = llhist_try[end]
    n_iters  = length(llhist_try)

    @printf("seed=%d  iters=%d  final log-likelihood=%.6f\n",
            sd, n_iters, ll_try)

    if ll_try > best_ll
        best_ll = ll_try
        best = (θ_try, π_try, A_try, llhist_try, sd)
    end
end


    θ, π, A, llhist, best_seed = best
    return θ, π, A, llhist, best_seed, best_ll
end

# Main script

DATA_PATH = joinpath(@__DIR__, "meteo1.csv")

@printf("Script directory (@__DIR__) = %s\n", @__DIR__)
@printf("Looking for data at        = %s\n", DATA_PATH)
@assert isfile(DATA_PATH) "meteo1.csv not found. Put it in the same folder as this script: $(@__DIR__)"

df = CSV.read(DATA_PATH, DataFrame; header=false)
X = Matrix{Int}(df)

# Run EM with multi-start (best practice)
seeds = 1:10
θ, π, A, llhist, best_seed, best_ll = em_multistart(
    X; K=3, S=3, max_iter=200, tol=1e-6, eps=1e-12, seeds=seeds, verbose_each=false
)

@printf("\nSelected run: seed=%d, final log-likelihood=%.6f\n\n", best_seed, best_ll)

println("Mixture weights θ (k=1..3):")
println(θ)
println()

println("Initial distributions π_k over states 0,1,2:")
for k in 1:3
    @printf("k=%d: [%.6f, %.6f, %.6f]\n", k, π[k,1], π[k,2], π[k,3])
end
println()

println("Transition matrices A_k (rows from-state 0,1,2; cols to-state 0,1,2):")
for k in 1:3
    println("k=$k:")
    for s in 1:3
        @printf("  from %d: [%.6f, %.6f, %.6f]\n", s-1, A[k,s,1], A[k,s,2], A[k,s,3])
    end
    println()
end

# Posterior for first 10 sequences
γ = responsibilities(X, θ, π, A)
m = min(10, size(X,1))
println("Posterior station probabilities for the first $m sequences:")
for n in 1:m
    @printf("row %d: P(k=1|x)=%.6f, P(k=2|x)=%.6f, P(k=3|x)=%.6f\n",
            n, γ[n,1], γ[n,2], γ[n,3])
end
