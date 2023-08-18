using LinearAlgebra


struct TuningCurve
    f::Function
    max_rate::Float32
    argmax_rate::Float32
    σ::Float32

    function TuningCurve(max_rate, argmax_rate, σ)
        f(s) = max_rate * exp(-0.5 * ((s - argmax_rate) / σ)^2)
        new(f, max_rate, argmax_rate, σ)
    end
end


mutable struct SpikeTrain
    """Struct representation of a spike train with a firing rate that is a 
    non-constant function of time.

    Parameters
    ----------
    T : number > 0 
        Duration of the trial.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain.
    tuncurve : TuningCurve
        The TuningCurve struct represents the function that governs the 
        variation of r = f(s), the firing rate as a function of a stimulus 
        parameter. The stimulus parameter is on its turn  a function s = g(t) 
        of time.
    stimcurve : function
        A periodic mapping from ℝ to ℝ that is the function s(t) of a stimulus
        parameter with respect to time. Periodicity must be imposed into
        whatever function describes the stimulus across time for some of the
        calculations to make sense. Such condition is often easy to impose
        using standard mathematical methods even on non-periodic functions.

    Fields
    ----------
    T : Float32 > 0 
        Duration of the trial.
    Δt : Float32 > 0
        Size of the infinitesimally small bins that split the time domain.
    t : Vector{Float32}
        Vector representation of the time domain [0, T] where t_i is the ith 
                            
                                [t_i, t_i + Δt]
    spike_train : Vector{Int} 
        A binary vector whose ith value is ρ(tᵢ).
    tuncurve : TuningCurve. 
        The TuningCurve instance that represents the function that governs the 
        variation of r = f(s), the firing rate as a function of a stimulus 
        parameter. The stimulus parameter is on its turn  a function s = g(t) 
        of time, where g is the stimulus curve.
    stimcurve : f : ℝ → ℝ
        A periodic mapping from ℝ to ℝ that is the function s(t) of a stimulus
        parameter with respect to time. Periodicity must be imposed into
        whatever function describes the stimulus across time for some of the
        calculations to make sense. Such condition is often easy to impose
        using standard mathematical methods even on non-periodic functions.
    """

    T::Float32
    Δt::Float32
    t::Vector{Float32}
    n::Int
    spike_train::Vector{Int}
    avg_spike_train::Vector{Float32}
    spike_times::Vector{Float32}
    avg_n::Float32
    tuncurve::TuningCurve
    stimcurve::Function

    function SpikeTrain(T, Δt, tuncurve, stimcurve)
        δ(x) = x >= rand() ? 1 : 0
        t = collect(0:Δt:T)
        f = tuncurve.f
        g = stimcurve
        spikes = [δ(f(g(x)) * Δt) for x in t]
        spike_times = [t[index] for (index, value) in enumerate(spikes) if value == 1]
        n = length(spike_times)

        new(T, Δt, t, n, spikes, [], spike_times, 0, tuncurve, stimcurve)
    end
end

struct SpikeMatrix
    """
    Ω is a struct whose main attribute is a matrix of the same name. 
    The matrix Ω is an m × m matrix whose column vectors ω₁, …, ωₘ are binary
    vectors with simulated spike trains. The integer m is defined to be the 
    number of discrete time bins of which is simulation is composed. In 
    particular, m = T/Δt, where T is the abstract duration of the trial and 
    Δt the size of each discrete time bin. 

    Parameters 
    ----------
    T : number > 0 
        Duration of each simulation.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain on 
        each simulation.
    tuncurve : TuningCurve
        The TuningCurve of each simulation. 
        The TuningCurve struct represents the function that governs the 
        variation of r = f(s), the firing rate as a function of a stimulus 
        parameter. The stimulus parameter is on its turn  a function s = g(t) 
        of time.
    stimcurve : function
        The stimulus curve of each simulation. 
        It is a periodic mapping from ℝ to ℝ. It is the function s(t) of a stimulus
        parameter with respect to time. Periodicity must be imposed into
        whatever function describes the stimulus across time for some of the
        calculations to make sense. Such condition is often easy to impose
        using standard mathematical methods even on non-periodic functions.

    Fields 
    ------
    X : A square binary matrix.
    eigenvalues : Eigenvalus of Ω
    eigenvectors : Eigenvectors of Ω
    D : Diagonal representation of Ω over the basis formed by the eigenvectors.
    """

    X::Matrix
    eigenvalues::Vector{Number}
    eigenvectors::Matrix{Number}
    D::Diagonal
    t::Vector{Float32}

    function SpikeMatrix(T, Δt, tuncurve, stimcurve)
        m = T / Δt + 1
        binary_matrix = transpose(hcat([SpikeTrain(T, Δt, tuncurve, stimcurve).spike_train for _ in 1:m]...))
        eigen_decomp = eigen(binary_matrix)
        eigenvalues = eigen_decomp.values
        eigenvectors = eigen_decomp.vectors
        D = Diagonal(eigenvalues)
        t = collect(0:Δt:T)

        new(binary_matrix, eigenvalues, eigenvectors, D, t)
    end
end


function sta(spike_train::SpikeTrain)
    """
    The spike-triggered average (STA) of a stimulus is a function C(τ) of 
    time that computes the average value of a stimulus τ time units prior 
    to the generation of an action potential. For an individual spike train,
    it is defined as 

                            C(τ) = 1/n * ∑ⁿ (s(tᵢ - τ))

    where t₁, …, tₙ are the spike times in the train and s(t) is 
    the stimulus curve.
    """

    values = []
    for τ in spike_train.t
        Cτ = sum([spike_train.stimcurve(t - τ)
                  for t in spike_train.spike_times]
        ) / spike_train.n
        push!(values, Cτ)
    end

    return values
end

function mean_sta(st::SpikeTrain)
    """
    The mean spike-triggered average (STA) of a stimulus is the trial
    average of a  function C(τ) of time that computes the average value of
    a stimulus τ time units prior to the generation of an action potential.

    Because trial average spikes are no longer binary values in {0, 1} 
    but frequencies, there is no sequence {t₁, …, tₙ} to be used for the 
    computation, and the alternative definition

                            C(τ) = 1/⟨n⟩ ∫ᵀ ⟨ρ(t)⟩s(t - τ) dt 

    is used, where ⟨ ⋅ ⟩ denotes a trial average value and the inferior 
    integral bound is 0. This definition is a bit more complicated, and 
    its computation is more costly compared to STA estimation with a unique 
    spike train, but it is also statistically more precise.
    """

    values = []
    avg_st = st.avg_spike_train
    n = st.avg_n

    for τ in st.t
        c_of_τ = 0
        for time_index in range(1, length(st.t))
            t = st.t[time_index]
            ρ_at_t = avg_st[time_index]
            c_of_τ += ρ_at_t * st.stimcurve(t - τ)
        end
        push!(values, c_of_τ / n)
    end

    return values
end


function firing_rate(spike_train::SpikeTrain, t)
    """
    A time dependent definition of the firing rate: 

        1/Δt * ∫ ⟨ρ(τ) dτ 

        where the lower and upper integral bounds are t and t + Δt, 
        respectively.

    Args:
        t: Time-point at which to estimate the firing rate.
    """
    index = findall(x -> x == t, spike_train.t)
    avg_ρ_of_t = spike_train.avg_spike_train[index]
    return 1 / spike_train.Δt * avg_ρ_of_t
end

function firing_rate(spike_train::SpikeTrain)
    firing_rate.(spike_train.t)
end


function stim_fr_corr(spike_train::SpikeTrain, τ::Int)
    """Compute the time correlation of functions f and t. Importantly, 
    f and g are taken to be lists with the function values rather than 
    functions themselves, and τ is taken to be an indexing distance rather 
    than a real-valued distance.

    Args:
        f (list): Values of f(t)
        g (list): Values of g(t)
        τ (int): Index distance of the values being correlated at each time
        point.
    """

    r(t) = firing_rate(spike_train, t)
    s = spike_train.stimcurve
    corrs = [r(t) * s(t + τ) for t in spike_train.t]

    return sum(corrs) / spike_train.T
end

function stim_fr_corr(spike_train::SpikeTrain)
    return [stim_fr_corr(spike_train, t) for t in spike_train.t]
end

function gen_avg_spike_train(st::SpikeTrain, n_trials::Int)
    @assert n_trials > 0
    trials = [SpikeTrain(st.T, st.Δt, st.tuncurve, st.stimcurve) for _ in range(1, n_trials)]
    st.avg_spike_train = reduce(+, [trial.spike_train for trial in trials]) ./ n_trials
    st.avg_n = sum(trial.n for trial in trials) ./ n_trials
end

function get_avg_firing_rate(st::SpikeTrain)
    @assert !isempty(st.avg_spike_train)
    st.avg_n / st.T
end

function autocorr(st::SpikeTrain, τ::Int)
    r = get_avg_firing_rate(st)
    N = length(st.t)
    avg_st = st.avg_spike_train

    value = 0
    for t in range(1, N)
        t_plus_tau = t + τ <= N ? t + τ : ((t + τ) % N) + 1 # +1 to make index > 0
        a = avg_st[t] - r
        b = avg_st[t_plus_tau]
        value += a * b
    end
    value / st.T
end

function autocorr(st::SpikeTrain)
    [autocorr(st, τ) for τ in range(1, length(st.t))]
end


function ω(st::SpikeTrain, Δt::Int)
    """Approximate r(t) using the window function 
        
                        {  1/Δt     t ∈ [-Δt/2, Δt/2]
                w(t) = -{
                        {  0        otherwise

    Algorithmically, the function iteratively takes slices of ρ(t) of size 
    2 * Δt (or less out of bounds) and computes the average number of spikes 
    for each slice. Returns a list of length equal to the length of self.t 
    the time dimension with each average.

    Mathematically, this function slides a window of size 2 * Δt across
    the time domain. At each point t the window is centered at t. 
    w(t) is evaluated and thus the average spike count is computed for 
    each window.

    Args
        Δt (int > 0)   Half the number of time-units comprised by each 
                        window.
    """

    N = length(st.spike_train)
    counts = []
    for i in range(1, N)
        lower_bound = i > Δt ? i - Δt : 1
        upper_bound = i + Δt <= N ? i + Δt : N
        bin = st.spike_train[lower_bound:upper_bound]
        push!(counts, count(x -> x == 1, bin) / Δt)
    end
    return counts
end


function γ(st::SpikeTrain, σ::Float64)
    """Approximate r(t) using the window function 
        
                w(t) = 1/(√(2π)σ) ⋅ exp(- τ²/ (2σ²))

    where σ is a parameter that governs the intensity with which 
    distant spikes affect the value at t (analogous to Δt on the 
    delta filter).

    Algorithmically, the function computes at each point t in self.t the 
    value of w(t - s) for every spike-time s. A list of equal length as 
    t is returned with these values.

    Mathematically, this function slides a Gaussian window of σ amplitude 
    across the time domain. At each point t the window is centered at t and 

                        ∫ w(τ)ρ(t - τ) dτ

    is computed with integral bounds 0 and T. 

    Args
        σ (float > 0) Amplitude of the Gaussian window.
    """

    N = length(st.spike_train)
    g(t) = (1 / (sqrt(2 * pi) * σ)) * exp(-t^2 / (2 * σ^2))

    results = []
    for i in range(1, N)
        distances = [st.t[i] - x for x in st.spike_times]
        r_of_t = sum([g(d) for d in distances])
        push!(results, r_of_t)
    end

    return results
end




