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

    def gen_spike_train(self):
        pass

    def plot_spike_train(self, x_size=15, y_size=5, save=False):
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        res = sns.lineplot(x=self.t, y=self.spike_train, ax=ax)
        if save:
            res.get_figure().savefig("spike_trainb.png")
        return res

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
        g = lambda t:  (1/(math.sqrt(2* math.pi) * σ)) * np.exp(-t**2/(2*σ**2))

        spike_times = [self.t[index]
                         for index, value in enumerate(self.spike_train)
                         if value == 1]


        results = []
        for i in range(N):
            distances = [self.t[i] - x for x in spike_times]
            r_of_t = sum([g(d) for d in distances])
            results.append(r_of_t)
        
        return results



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
    stimcurve : ndarray 
        An array representing the values of s(t) the stimulus parameter over
        time. It has the same length as t.

    Methods
    -------
    f(s):
        The tuning curve function.
    """

    def __init__(self, T, Δt, tuncurve):
        super().__init__(T, Δt) 
        self.tuncurve = tuncurve
        self.stimcurve = np.arange(tuncurve.range[0],
                            tuncurve.range[1] + self.Δt, 
                           (tuncurve.range[1] - tuncurve.range[0])/999)

    def gen_spike_train(self):
        δ = lambda x: 1 if x >= random.random() else 0
        ρ = [δ(self.tuncurve.f(self.stimcurve[x]) * self.Δt) for x in range(len(self.t))] 
        return(ρ)


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



#def gen_spike_train(duration, bin_size, 
#                    r_max=2, s_max=0, sigma=2):
#    δ = lambda x: 1 if x >= random.random() else 0
#    t = np.arange(0, duration + bin_size, bin_size)
#    print(t)
#    s_of_t = np.arange(-40, 40 + bin_size, 80/999)
#    print(s_of_t)
#    ρ = [δ( tuning_curve(s_of_t[x], r_max, s_max, sigma) * bin_size) for x in range(len(t))] 
#    print(ρ)
#    res = sns.lineplot(x=t, y=ρ, ax=ax)
#    res.get_figure().savefig("spike_train.png")
#
#
#def tuning_curve(s, r_max, s_max, sigma):
#    return r_max * math.exp(-0.5 * ((s - s_max)/sigma)**2)






