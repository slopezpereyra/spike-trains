using Plots 

const equilibrium_potential = -65
const membrane_resistance = 10
const external_current = 2
const V₀ = -65
const Vᵣ = -65
const threshold = -50
const τ = 10
const dt = 0.1

const Eₖ = -70
const τₛ = 2
const rₘ = 0.1
const Δg = 40

function gsra(t, gₛ)
    if t == 0 
        return gₛ
    end 

    gₛ * exp(-t/τₛ)
end

function V(t, gₛ)
    if t == 0
        return(V₀)
    end 
    u = equilibrium_potential + membrane_resistance * external_current + rₘ * gsra(t, gₛ)*Eₖ
    w = -rₘ*gsra(t, gₛ) - 1
    
    (V(0, gₛ)*exp(t*w/τ)-u)/w
end

function sim(T, dt, gₛ)
    v = V₀
    values = []
    t = 0
    for i in 0:dt:T
        if v == 1 
            v = Vᵣ  
            push!(values, v)
            t = 0
            gₛ += Δg
            continue
        end
        if v > threshold 
            v = 1 
            push!(values, v)
            continue
        end
        v = V(t, gₛ) 
        t += dt
        push!(values, v)
    end
    plot(0:dt:T, values, xlabel="Time", ylabel="Membrane Potential", label="Membrane Potential")
end

sim(100, 0.1, 100)

