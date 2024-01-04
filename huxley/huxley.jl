using Plots 

function hh(s, e, T)
    gmax = [36, 120, 0.3] 
    E = [-77, 50, -54.387]
    I_ext = 0; V = -10; x = [0, 0, 1]; dt = 0.01
    x_plot = []; y_plot = [];
    y_plot_n = []
    y_plot_m = []
    y_plot_h = []
    y_plot_i = []
    for t in range(-30, T, step=dt)
        # Turn external current on or off according to time.
        if t == s  I_ext = 10 end
        if t == e I_ext = 0 end

        # Alpha functions 
        αₙ = 0.01*(V+55)/(1 - exp(-0.1*(V+55)))
        αₘ = 0.1*(V+40)/(1 - exp(-0.1*(V+40)))
        αₕ = 0.07 * exp(-0.05*(V+65))
        α = [αₙ, αₘ, αₕ]

        # Beta functions
        βₙ = 0.125 * exp(-0.0125*(V + 65))
        βₘ = 4 * exp(-0.0556*(V + 65) )
        βₕ = 1 / (1 + exp(-0.1*(V + 35)) )
        β = [βₙ, βₘ, βₕ]

        # τₓ and x₀, where x₀ is an expression needed to compute 
        # the activation variables
        τₓ = 1 ./ (α + β)
        x₀ = α .* τₓ

        # Numerical integration of activation variables
        x = x₀ + exp.(-dt./τₓ) .* (x .- x₀)

        # Compute conductances 
        g = [gmax[1] * x[1]^4, gmax[2] * x[2]^3*x[3], gmax[3]]

        # Ohm's law 
        I = sum(g .* (V .- E))

        # Update voltage (membrane potential) 
        V += dt * (I_ext - I)

        if t >= 0 
            push!(x_plot, t)
            push!(y_plot, V)
            push!(y_plot_i, I)
            push!(y_plot_n, x[1])
            push!(y_plot_m, x[2])
            push!(y_plot_h, x[3])
        end 
    end
    p1 = plot(x_plot, y_plot, label="")
    ylabel!("V")
    p2 = plot(x_plot, y_plot_n, label="")
    ylabel!("n")
    p3 = plot(x_plot, y_plot_m, label="")
    ylabel!("m")
    p4 = plot(x_plot, y_plot_h, label="")
    ylabel!("h")
    p5 = plot(x_plot, y_plot_i, label="")
    ylabel!("I")
    plot(p1, p2, p3, p4, p5, layout=(4, 2))
    xlabel!("t")
end

hh(10, 200, 250)
