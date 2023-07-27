using Plots

function ΔVₘ(w::Number, t::Number, peak::Number)
    """Non-NMDA neuron EPSP and IPSP basic model. 
    
    Observe that ΔVₘ measures change and hence its output is 
    a relative PSP. (Relative to the membrane potential of the 
    postsynaptic neuron at the moment the excitatory signal is 
    received.)

    The time required for such signal to be received (neurotransmitter
    liberation, difussion of neurotransmitters, opening of the ion gates)
    is not included in the model.
    """
    w * t * exp(-t/peak)
end

function hh()
    
    # Maximual conductances for K, Na, R.
    gmax = [36, 120, 0.3]

    # Battery voltages, equilibrium potentials
    E = [-12, 115, 10.613]
    
    # I_ext is the external current; 
    # x holds the n, m, h variables (conductances of 
    # the different ion channels). 
    I_ext = 0; V = -10; x = [0, 0, 1];
    dt = 0.01
    x_plot = []; y_plot = [];

    for t in range(-30, 250, step=dt)
        # Turn external current on or off according to time.
        if t == 10  I_ext = 10 end
        if t == 200 I_ext = 0 end

        # Alpha functions 
        α₁ = (10 - V)/(100 * (exp((10 - V)/10) - 1))
        α₂ = (25 - V) / (10 * (exp((25-V)/10) - 1)) 
        α₃ = 0.07 * exp(-V/20)
        α = [α₁, α₂, α₃]

        # Beta functions
        β₁ = 0.125 * exp(-V/80)
        β₂ = 4 * exp(-V/18)
        β₃ = 1 / (exp((30 - V)/10) + 1)
        β = [β₁, β₂, β₃]

        # τₓ and x₀
        τₓ = 1 ./ (α + β)
        x₀ = α .* τₓ

        # Leaky integration with Euler method.
        # See formula above.
        x = (1 .- dt ./ τₓ) .* x + dt ./ τₓ .* x₀

        # Compute conductances 
        g = [gmax[1] * x[1]^4, gmax[2] * x[2]^3*x[3], gmax[3]]

        # Ohm's law 
        I = g .* (V .- E)

        # Update voltage (membrane potential) 
        V += dt * (I_ext - sum(I))

        if t >= 0 
            push!(x_plot, t)
            push!(y_plot, V)
        end 
    end
    plot(x_plot, y_plot, label="Membrane potential")
    xlabel!("t")
    ylabel!("V")
end

hh()

x = range(0, 10, length=100)
epsp = ΔVₘ.(1, x, 3)
ipsp = ΔVₘ.(-2, x, 1)
# Membrane potential prior to excitatory signal 
mp = repeat([0], 100)


plot(x, epsp, label="EPSP", lw=2)
plot!(x, ipsp, label="IPSP", lw=2)
plot!(x, mp, label=["MP prior to signal"], ls=:dash, color=:black)
xlabel!("t")
ylabel!("Relative PSP")
