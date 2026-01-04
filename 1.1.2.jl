#H2: two explosions
Lmax = maximum(loglik)
w = exp.(loglik .- Lmax)

logp_H2 = Lmax + log(sum(w)) - 2 * log(S)



#H1: one explosion
loglik1 = zeros(S)

for s in 1:S
    μ = F[s, :]                    
    r = v_obs .- μ
    loglik1[s] = -sum(r.^2) / (2 * sigma^2)
end

L1max = maximum(loglik1)
w1 = exp.(loglik1 .- L1max)

logp_H1 = L1max + log(sum(w1)) - log(S)



#Final answer
println("log p(v|H2) - log p(v|H1) = ", logp_H2 - logp_H1)
