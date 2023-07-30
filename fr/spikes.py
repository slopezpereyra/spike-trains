import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


class SpikeTrain : 
    """Class representation of a spike train.
    
    The class contains the attributes and methods necessary for the simulation
    of artificial spike trains in accordance to current neuroscientific
    understanding.

    A spike train is a function ρ(t) and over a constant sequence of time
    events

                            {t_1, ..., t_n} 

    such that 

                            ρ(t) = ∑ⁿδ(t - t_i)

    where δ is Dirac's delta. We refer to this function as the neural response
    function. It appropriately describes the existence or non-existence of
    action potentials during a trial of finite duration. A fundamental property
    that derives from Dirac's delta is 

                            ρ(t) = ∫ ρ(t) dt

    where the integral bounds are negative and positive infinity. Because the
    trial is finite this bounds are replaced by 0 and T, where T is the dura-
    tion of the trial. The integral can also be taken under any arbitrarily
    small interval [t, t + Δt] - and much of the mathematics involved in the
    simulation operates over such indefinitely small time frames. 


    Parameters
    ----------
    T : number > 0 
        Duration of the trial.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain.
    sim_on_init : bool 
        Run the simulation on instantiation?

    Attributes
    ----------
    T : number > 0 
        Duration of the trial.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain.
    t : ndarray
        Array representing the time domain [0, T] where t_i is the ith 
                            
                                [t_i, t_i + Δt]
    spike_train : list 
        The value of the neural response ρ(t) at each point in the simulation.
    spike_times : list 
        A list with the time of occurrence of each spike - a representation of 
        the sequence {t₁, …, tₙ}
        
    n : int 
        The number of spikes in the train
    avg_spike_train : list 
        A list representing ⟨ρ(t)⟩, the average value of the neural response
        function at each point in the simulation. Importantly, ⟨ρ⟩ is an
        arbitrarily precise approximation of the firing rate, and each 
        ⟨ρ(t)⟩ represents the probability that a spike occurs at time t.

    n_avg : float 
        The average number of spikes in the series of simulations that 
        produced the avg_spike_train.

    Methods
    -------
    f(s):
        The tuning curve function.
    """

    def __init__(self, T, Δt):
        self.T = T
        self.Δt = Δt
        self.t = np.arange(0, self.T + self.Δt, self.Δt)
        self.spike_train = []
        self.spike_times = []
        self.n = 0
        self.avg_spike_train = []
        self.n_avg = 0

    def gen_spike_train(self):
        pass

    def plot_spike_train(self, x_size=15, y_size=5, save=False):
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        res = sns.lineplot(x=self.t, y=self.spike_train, ax=ax)
        if save:
            res.get_figure().savefig("spike_trainb.png")
        return res

    def get_spike_times(self, ρ):
        """Wrapper for getting the spike times of a given spike train.

        Args:
            ρ : The spike train whose spike times to get.
        """
        
        spike_times = [self.t[index]
                         for index, value in enumerate(ρ)
                         if value == 1]
        return(spike_times)

    def linear_delta_filter(self, Δt):
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

        Args:
            Δt (int > 0):   Half the number of time-units comprised by each 
                            window.
        """

        N = len(self.spike_train)
        counts = []
        for i in range(N):
            lower_bound = i - Δt if i > Δt else 0
            upper_bound = i + Δt if i + Δt <= N - 1 else N - 1
            bin = self.spike_train[lower_bound:upper_bound]
            counts.append(bin.count(1)/Δt)

        return counts


    def gaussian_filter(self, σ):
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

        Args:
            σ (float > 0): Amplitude of the Gaussian window.
        """

        N = len(self.spike_train)
        g = lambda t:  (1/(math.sqrt(2* math.pi) * σ)) * np.exp(-t**2/(2*σ**2))

        results = []
        for i in range(N):
            distances = [self.t[i] - x for x in self.spike_times]
            r_of_t = sum([g(d) for d in distances])
            results.append(r_of_t)
        
        return results

    def generate_avg_spike_train(self, trials):
        """Generates the trial-average spike train.

        As trials tends to infinity, the average spike train ⟨ρ(t)⟩ approachs
        r(t) the time-dependent firing rate and hence, for sufficiently large
        trials, this quantities are interchangable.

        Args:
            trials (int): number of trials to average out
        """
        counts = []
        ρ = self.gen_spike_train()
        counts.append(len(self.get_spike_times(ρ)))
        for i in range(trials - 1):
            ρᵢ = self.gen_spike_train()
            ρ = [x + y for x, y in zip(ρ, ρᵢ)]
            counts.append(len(self.get_spike_times(ρᵢ)))

        ρ_avg = [x/trials for x in ρ]
        self.n_avg = sum(counts)/trials
        return ρ_avg

class NonConstantSpikeTrain (SpikeTrain) : 
    """Class representation of a spike train with a firing rate that is a 
    non-constant function of time.
    

    Parameters
    ----------
    T : number > 0 
        Duration of the trial.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain.
    sim_on_init : bool 
        Run the simulation on instantiation?
    tuncurve : TuningCurve. 
        The TuningCurve object represents the function that governs the 
        variation of r = f(s), the firing rate as a function of a stimulus 
        parameter. The stimulus parameter is on its turn  a function s = g(t) 
        of time.

    Attributes
    ----------
    T : number > 0 
        Duration of the trial.
    Δt : number > 0
        Size of the infinitesimally small bins that split the time domain.
    t : ndarray
        Array representing the time domain [0, T] where t_i is the ith 
                            
                                [t_i, t_i + Δt]
    spike_train : list 
        The value of the neural response ρ(t) at each point in the simulation.
    tuncurve : TuningCurve. 
        The TuningCurve object represents the function that governs the 
        variation of r = f(s), the firing rate as a function of a stimulus 
        parameter. The stimulus parameter is on its turn  a function s = g(t) 
        of time.
    stimcurve : f : ℝ → ℝ
        A periodic mapping from ℝ to ℝ that is the function s(t) of a stimulus
        parameter with respect to time. Periodicity must be imposed into
        whatever function describes the stimulus across time for some of the
        calculations to make sense. Such condition is often easy to impose
        using standard mathematical methods even on non-periodic functions.

    Methods
    -------
    sta:
        Computes the spike-triggered average (STA) of the stimulus.
        The spike-triggered average (STA) of a stimulus is a function C(τ) of 
        time that computes the average value of a stimulus τ time units prior 
        to the generation of an action potential. For an individual spike train,
        it is defined as 

                                C(τ) = 1/n * ∑ⁿ (s(tᵢ - τ))

        where t₁, …, tₙ are the spike times in the train and s(t) is 
        the stimulus curve.
    mean_sta: 
        Generates the mean spike-triggered average (MSTA) of a stimulus. This is
        the trial average of the STA function.

        Because trial average spikes are no longer binary values in {0, 1} but
        frequencies, there is no sequence {t₁, …, tₙ} to be used for the
        computation, and the alternative definition

                                C(τ) = 1/⟨n⟩ ∫ᵀ ⟨ρ(t)⟩s(t - τ) dt 

        is used, where ⟨ ⋅ ⟩ denotes a trial average value and the inferior
        integral bound is 0. This definition is a bit more complicated, and its
        computation is more costly compared to STA estimation with a unique
        spike train, but it is also statistically more precise.

    """

    def __init__(self, T, Δt, tuncurve, stimcurve):
        super().__init__(T, Δt) 
        self.tuncurve = tuncurve
        self.stimcurve = stimcurve

    def gen_spike_train(self):
        """Generates a spike train whose firing rate is time- and stimulus- 
        dependent.

        A spike train is a function that is 0 everywhere except at a particular
        sequence of time points {t₁, …, tₙ}, where it is one. It represents the
        activations of a neuron in time. The occurrence of non-zero values, or
        action potentials, is governed by the influence of a stimulus across
        time. The susceptibility of the neuron to different parameters of the
        stimulus is determined by the tuning curve. The fluctuation of stimulus
        values across times is determind by the stimulus curve.

        As a statistical process, the generation of action potentials in a
        spike train is a Poisson process. If the firing rate were constant the
        process would be governed by a λ = -rT parameter, with T the total
        duration of the trial and r a constant firing rate. This assumption 
        doesn't hold in reality and spike-generation is a non-homogeneous 
        Poisson process. The probability density for the occurrence of n spikes 
        in a trial of length N is 

                                exp(  -∫ᵀr(t)dt  ) Πⁿ r(tᵢ)

        where r(t) is the time-dependent firing rate. The values of r(t) can be 
        approximated with the linear filters provided in this class.

        """
        δ = lambda x: 1 if x >= random.random() else 0
        f, g = self.tuncurve.f, self.stimcurve
        ρ = [δ(f(g(x)) * self.Δt) for x in self.t] 
        self.spike_times = [self.t[index]
                         for index, value in enumerate(ρ)
                         if value == 1]
        self.n = len(self.spike_times)
        return(ρ)

    def sta(self):
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
        for τ in self.t:
            Cτ = sum([self.stimcurve(t - τ) for t in self.spike_times])/self.n
            values.append(Cτ)

        return values

    def mean_sta(self):
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
        spike_train = self.avg_spike_train 
        n = self.n_avg 

        for τ in self.t:
            c_of_τ = 0
            for time_index in range(len(self.t)):
                t = self.t[time_index]
                ρ_at_t = spike_train[time_index]
                c_of_τ += ρ_at_t * self.stimcurve(t - τ)
            values.append(c_of_τ/n)

        return values
        
        


class TuningCurve:
    """Class representation of a tuning curve.

    A tuning curve is a function r = f(s) where r is the firing rate and 
    s is a stimulus parameter. 

    Parameters
    ----------
    r_max : number 
        Maximum value of f(s) - represents the highest firing rate elicited 
        by the stimulus.
    s_max : number
        Value of s that is the argmax of f. It is the value of the parameter 
        that elicits r_max the highest firing rate.
    σ : number
        Value that determines the rate at which the elicited firing rate decays 
        as s moves away from s_max.

    Attributes
    ----------
    r_max : number 
        Maximum value of f(s) - represents the highest firing rate elicited 
        by the stimulus.
    s_max : number
        Value of s that is the argmax of f. It is the value of the parameter 
        that elicits r_max the highest firing rate.
    σ : number
        Value that determines the rate at which the elicited firing rate decays 
        as s moves away from s_max.

    Methods
    -------
    f(s):
        The tuning curve function.
    """
    
    def __init__(self, r_max, s_max, σ, range):
        self.r_max = r_max 
        self.s_max = s_max
        self.σ = σ
        self.range = range

    def f(self, s):
        return self.r_max * math.exp(-0.5 * ((s - self.s_max)/self.σ)**2)

