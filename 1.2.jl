#  Q2 Meeting scheduling
# Part (a)

using Distributions

function cdf_punctual(t::Real)
    if t <= 0
        return 0.7
    elseif t <= 5
        return 0.8
    elseif t <= 10
        return 0.9
    elseif t <= 15
        return 0.97
    elseif t <= 20
        return 0.99
    else
        return 1.0
    end
end

function meeting_time_punctual(N::Int; target = 0.9)
    # Only these points matter because the CDF jumps here.
    candidates = [0, 5, 10, 15, 20]

    for T0 in candidates
        p_all_on_time = cdf_punctual(T0)^N
        if p_all_on_time >= target
            return T0, p_all_on_time
        end
    end

    return nothing
end

println("===== Part (a): punctual model =====")
for N in (3, 5, 10)
    T0, prob = meeting_time_punctual(N)
    println("N = $N:")
    println("  T0 = $T0 minutes early")
    println("  P(catching train) = $(round(prob, digits=4))")
end



# Part (b)


function cdf_not_punctual(t::Real)
    if t <= 0
        return 0.5
    elseif t <= 5
        return 0.7
    elseif t <= 10
        return 0.8
    elseif t <= 15
        return 0.9
    elseif t <= 20
        return 0.95
    else
        return 1.0
    end
end

# Prior over Z_i:
const w_punctual     = 2/3
const w_not_punctual = 1/3

# Mixture CDF: F_mix(t) = P(D ≤ t)
function cdf_mixture(t::Real)
    return w_punctual * cdf_punctual(t) +
           w_not_punctual * cdf_not_punctual(t)
end

println("\n===== Part (b): mixture model =====")
for N in (3, 5, 10)
    # Use the same T0 from part (a)
    T0, _ = meeting_time_punctual(N)

    F_mix  = cdf_mixture(T0)
    p_catch = F_mix^N
    p_miss  = 1 - p_catch

    println("N = $N:")
    println("  Using T0 = $T0 minutes early (from part (a))")
    println("  P(missing  train) = $p_miss")
    println(F_mix)
    println(p_catch)
end


# Bonus：N=5 

println("\n===== Bonus: posterior for number of non-punctual friends (N = 5) =====")

N_bonus = 5
T0_bonus, _ = meeting_time_punctual(N_bonus)  # should be 20 minutes

# CDF at 20 min for each type
cdf20_p = cdf_punctual(T0_bonus)      # = 0.99
cdf20_n = cdf_not_punctual(T0_bonus)  # = 0.95

p_late_p = 1 - cdf20_p   # punctual friend late  (> 20)
p_late_n = 1 - cdf20_n   # non-punctual friend late (> 20)

# Prior for K ~ Binomial(N, p_notpunctual)
p_notpunctual = w_not_punctual
binom_prior = Binomial(N_bonus, w_not_punctual)

# Likelihood: P(miss train | K = k)
function likelihood_miss(k::Int)

    prob_all_on_time = (1 - p_late_n)^k * (1 - p_late_p)^(N_bonus - k)
    return 1 - prob_all_on_time
end

# Unnormalized posterior
posterior_unnorm = [
    pdf(binom_prior, k) * likelihood_miss(k) for k in 0:N_bonus
]

# Normalize
posterior = posterior_unnorm ./ sum(posterior_unnorm)

println("Posterior distribution P(K = k | missed train, N = 5, T0 = $T0_bonus):")
for k in 0:N_bonus
    println("  K = $k : ", round(posterior[k+1], digits = 4))
end