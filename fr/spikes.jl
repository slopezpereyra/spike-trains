using Distributions



function generate_spikes(rate, duration, bin_size)
    dist = Poisson(-rate*duration)
    f(x) = x >= rand(dist) ? 1 : 0
    spikes = [f(x) for x in range(0, duration, length=duration*bin_size)]
    print(spikes)
end

generate_spikes(2, 10, 1)
