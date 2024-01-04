using Plots 

const equilibrium_potential = -65
const membrane_resistance = 10
const external_current = 2
const V₀ = -65
const Vᵣ = -65
const threshold = -50
const τ = 10
const dt = 0.1

function V(t)
    if t == 0
        return(V₀)
    end
    u = equilibrium_potential + membrane_resistance * external_current
    u + (V(0) - u) * exp(-t/τ)
end

function sim(T, dt)
    v = V₀
    values = []
    t = 0
    for i in 0:dt:T
        if v == 1 
            v = Vᵣ  
            push!(values, v)
            t = 0
            continue
        end
        if v > threshold 
            v = 1 
            push!(values, v)
            continue
        end
        v = V(t) 
        t += dt
        push!(values, v)
    end
    plot(0:dt:T, values, xlabel="Time", ylabel="Membrane Potential", label="Membrane Potential")
end

sim(100, 0.1)

function V(t, Δt)
    if t == 0
        return(V₀)
    end
    u = equilibrium_potential + membrane_resistance * cos(t)sin(t)*100*randn()
    u + (V(t - Δt) - u) * exp(-Δt/τ)
end

function sim(T, dt)
    v = V₀
    values = []
    t = 0
    for i in 0:dt:T
        if v == 1 
            v = Vᵣ  
            push!(values, v)
            t = 0
            continue
        end
        if v > threshold 
            v = 1 
            push!(values, v)
            continue
        end
        v = V(t, dt) 
        t += dt
        push!(values, v)
    end
    plot(0:dt:T, values, xlabel="Time", ylabel="Membrane Potential", label="Membrane Potential")
end

sim(100, 0.1)
