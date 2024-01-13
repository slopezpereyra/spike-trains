import numpy as np
import matplotlib.pyplot as plt
from manim import *


GMAX = np.array([36, 120, 0.3])
E = np.array([-77, 50, -54.387])
EXT_START = 25
EXT_END = 150
V_GLOBAL = -10

ACTIVATION_VARS = {
        "n": 0,
        "m": 0,
        "h": 1
        }

def reset():

    global V_GLOBAL 
    global ACTIVATION_VARS 
    V_GLOBAL = -10
    ACTIVATION_VARS = {
            "n": 0,
            "m": 0,
            "h": 1
            }


def alpha(x):
    if x == "n":
        return 0.01*(V_GLOBAL +55)/(1 - np.exp(-0.1*(V_GLOBAL +55)))
    if x == "m":
        return 0.1*(V_GLOBAL +40)/(1 - np.exp(-0.1*(V_GLOBAL +40)))
    if x == "h":
        return 0.07 * np.exp(-0.05*(V_GLOBAL +65))

    raise ValueError("x must be n, m, or h")


def beta(x):
    if x == "n":
        return 0.125 * np.exp(-0.0125*(V_GLOBAL  + 65))
    if x == "m":
        return 4 * np.exp(-0.0556*(V_GLOBAL  + 65) )
    if x == "h":
        return 1 / (1 + np.exp(-0.1*(V_GLOBAL  + 35)) )

    raise ValueError("x must be n, m, or h")

def tau(x):
    return 1/(alpha(x) + beta(x))

def activation_var(x):
    global ACTIVATION_VARS
    τ_x = 1/(alpha(x) + beta(x))
    x_limit = alpha(x) * τ_x
    result = x_limit + np.exp(-0.01 / τ_x) * (ACTIVATION_VARS[x] - x_limit)
    ACTIVATION_VARS[x] = result
    return result

def current():
    x = np.array( [activation_var("n"), activation_var("m"), activation_var("h")] )
    g = np.array([GMAX[0] * x[0]**4, GMAX[1] * x[1]**3 * x[2], GMAX[2]])
    # Ohm's law
    return np.sum(g * (V_GLOBAL  - E))

def membrane_potential(t, which):
    global V_GLOBAL
    if t > EXT_START and t < EXT_END: 
        ext = 10 
    else:
        ext = 0
    I = current()
    V_GLOBAL += 0.01 * (ext - I)
    return float(which)


class MembranePotential(Scene):

    def construct(self):
        ax = Axes(
            x_range=[0, 200, 0.1],
            y_range=[-90, 70, 1],
            tips=True,
            x_axis_config={"include_numbers": False,
                        "numbers_to_include": [25, 150], 
                         "include_ticks": False
                         },
            y_axis_config={"include_numbers": False,
                        "numbers_to_include": E, 
                         "include_ticks": False
                         }
        )

        
        graph1 = ax.plot(lambda x: membrane_potential(x, V_GLOBAL), x_range=[-30, 200]).set_color(RED)
#        graph2 = ax.plot(lambda x: membrane_potential(x, ACTIVATION_VARS["m"]), x_range=[-30, 25]).set_color(BLUE)
#        graph3 = ax.plot(lambda x: membrane_potential(x, ACTIVATION_VARS["h"]), x_range=[-30, 25]).set_color(WHITE)
        self.add(ax)
        self.play(Create(graph1), run_time=10)
       # 
       # self.add(axes, graph)
       # self.wait(5)

