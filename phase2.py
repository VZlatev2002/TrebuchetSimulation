#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 12:12:00 2022

@author: tolgakaancelebi
"""

import Trebuchet_Functions as tf
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from sympy import symbols, Eq, solve

# System Parameters

g       = 9.81
m_B     = 12
M_CW    = 4000
M_P     = 100
L_B     = 8
H       = 6
L_S     = 6
L_BC    = 1.5
L_BP    = L_B - L_BC

p = [g, m_B, M_CW, M_P, L_B, H, L_S, L_BC, L_BP]

# Initial conditions


# Equation of motion

def f2(t, z):
    
    M2 = tf.M2fun(z[1],p)
    R2 = tf.R2fun(z[0],z[1],z[2],z[3],p)
    
    x, y = symbols('x y')
    eq1 = Eq(M2[0][0]*x + M2[0][1]*y + R2[0])
    eq2 = Eq(M2[1][0]*x + M2[1][1]*y + R2[1])
    
    sol_dict = solve((eq1,eq2), (x, y))

    return [z[2],
            z[3],
            sol_dict[x],
            sol_dict[y]]

# Event function

def event2(t,z):

    return z[1] - np.pi




#ODE Solver & initial conditions
def sim2(t1, z1):
    phi = tf.phifun(z1[0][-1],p)
    dphi = tf.dphifun(z1[0][-1], z1[1][-1],p)

    # Equation of motion
    event2.terminal = True
    event2.direction = 1
    rtol = 1e-6

    a = z1[0][-1]

    z0 = [z1[0][-1],
          phi,
          z1[1][-1],
          dphi]

# Execute IVP solver

    sol = solve_ivp(f2, (t1[-1],t1[-1]+10), z0, rtol=rtol, events=event2)
    t2 = sol.t
    z2 = sol.y
    
    return t2,z2