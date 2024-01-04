
import numpy as np
import matplotlib.pyplot as plt

def hh(s, e, T):
    gmax = np.array([36, 120, 0.3])
    E = np.array([-77, 50, -54.387])
    ext = 0
    V = -10
    x = np.array([0, 0, 1])
    x_plot = []
    y_plot = []
    y_plot_n = []
    y_plot_m = []
    y_plot_h = []
    y_plot_i = []
    dt = 0.01

    for t in np.arange(-30, T, dt):
        # Turn external current on or off according to time.
        if t > s:
            ext = 10
        if t > e:
            ext = 0

        # Alpha functions 
        αₙ = 0.01*(V+55)/(1 - np.exp(-0.1*(V+55)))
        αₘ = 0.1*(V+40)/(1 - np.exp(-0.1*(V+40)))
        αₕ = 0.07 * np.exp(-0.05*(V+65))
        α = np.array([αₙ, αₘ, αₕ])

        # Beta functions
        βₙ = 0.125 * np.exp(-0.0125*(V + 65))
        βₘ = 4 * np.exp(-0.0556*(V + 65) )
        βₕ = 1 / (1 + np.exp(-0.1*(V + 35)) )
        β = np.array([βₙ, βₘ, βₕ])

        # τₓ and x₀, where x₀ is an expression needed to compute
        # the activation variables
        τ_x = 1/(α + β)
        x_limit = α * τ_x

        # Numerical integration of activation variables
        x = x_limit + np.exp(-dt / τ_x) * (x - x_limit)

        # Compute conductances
        g = np.array([gmax[0] * x[0]**4, gmax[1] * x[1]**3 * x[2], gmax[2]])

        # Ohm's law
        I = np.sum(g * (V - E))

        # Update voltage (membrane potential)
        V += dt * (ext - I)

        if t >= 0:
            x_plot.append(t)
            y_plot.append(V)
            y_plot_i.append(I)
            y_plot_n.append(x[0])
            y_plot_m.append(x[1])
            y_plot_h.append(x[2])

    fig, axs = plt.subplots(4, 2, figsize=(12, 10))

    axs[0, 0].plot(x_plot, y_plot)
    axs[0, 0].set_ylabel("V")

    axs[1, 0].plot(x_plot, y_plot_n)
    axs[1, 0].set_ylabel("n")

    axs[2, 0].plot(x_plot, y_plot_m)
    axs[2, 0].set_ylabel("m")

    axs[3, 0].plot(x_plot, y_plot_h)
    axs[3, 0].set_ylabel("h")

    axs[0, 1].plot(x_plot, y_plot_i)
    axs[0, 1].set_ylabel("I")

    for ax in axs.flat:
        ax.set_xlabel("t")

#    plt.plot(x_plot, y_plot)
    plt.show()

hh(5, 15, 20)
