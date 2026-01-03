using DelimitedFiles
using Plots

include("earthquakeExerciseSetup.jl")
x, y, x_sensor, y_sensor = earthquake_exercise_setup()

S = length(x)              
N = length(x_sensor)       
sigma = 0.2                

data_path = joinpath(@__DIR__, "EarthquakeExerciseData.txt")
v_obs = vec(readdlm(data_path))
@assert length(v_obs) == N

#    F[k, i] = 1 / (d^2 + 0.1)
F = Array{Float64}(undef, S, N)

for k in 1:S
    for i in 1:N
        dx = x_sensor[i] - x[k]
        dy = y_sensor[i] - y[k]
        d2 = dx^2 + dy^2
        F[k, i] = 1.0 / (d2 + 0.1)
    end
end

#    loglik[k1, k2] = -∑(v_i - (F[k1,i]+F[k2,i]))^2 / (2σ^2)
loglik = zeros(Float64, S, S)

for k1 in 1:S
    f1 = @view F[k1, :]
    for k2 in 1:S
        f2 = @view F[k2, :]
        μ  = f1 .+ f2              
        r  = v_obs .- μ            
        loglik[k1, k2] = -sum(r.^2) / (2 * sigma^2)
    end
end

Lmax = maximum(loglik)
w = exp.(loglik .- Lmax)

unnorm_s1 = sum(w, dims = 2)          # 对第二维 (k2) 求和
posterior_s1 = vec(unnorm_s1 ./ sum(unnorm_s1))

println("Sum of posterior p(s1 | v)  = ", sum(posterior_s1))

#plot
graymap = cgrad([:white, :black])

plt = scatter(
    x, y;
    marker_z   = posterior_s1,
    ms         = 4,
    msw        = 0,              
    color      = graymap,
    colorbar   = true,
    xlabel     = "x",
    ylabel     = "y",
    title      = "Posterior p(s₁ | v)",
    leg        = false,
    aspect_ratio = 1,
)


scatter!(
    plt,
    x_sensor, y_sensor;
    m  = (:circle, 3),   
    msw = 0,             
    c  = :red,
    leg = false,
)


plot!(
    plt,
    [x_sensor; x_sensor[1]],    
    [y_sensor; y_sensor[1]],
    c = :black,
    lw = 1.5,
    leg = false,
)


max_len = 0.6
vals = v_obs ./ maximum(abs.(v_obs)) .* max_len

for i in 1:length(x_sensor)
    dirx = x_sensor[i] / hypot(x_sensor[i], y_sensor[i])
    diry = y_sensor[i] / hypot(x_sensor[i], y_sensor[i])

    x_start = x_sensor[i]
    y_start = y_sensor[i]

    len_i = vals[i]
    x_end = x_start + len_i * dirx
    y_end = y_start + len_i * diry

    plot!(
        plt,
        [x_start, x_end],
        [y_start, y_end];
        c  = :red,          
        lw = 1.5,
        leg = false,
    )
end

ind = argmax(w)   # 返回一个 CartesianIndex(k1, k2)
k1_map, k2_map = Tuple(ind)

println("Most likely pair of explosion points:")
println("s1: x = ", x[k1_map], ", y = ", y[k1_map])
println("s2: x = ", x[k2_map], ", y = ", y[k2_map])


display(plt)
savefig(plt, "posterior_s1_red_sensors.png")