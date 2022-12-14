#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:35:07 2022

@author: tolgakaancelebi
"""

import numpy as np
from scipy.integrate import solve_ivp


from Trebuchet_Functions import load_treb_data, animate_trebuchet, M1fun, R1fun, dv_Sfun

# System Parameters

g = 9.81
m_B = 12
M_CW = 4000
M_P = 100
L_B = 8
p_a = 1.225 # density of air
p_w = 1000 # density of water
C_d = 0.5 # drag coefficient
pi = np.pi
L_BC = 1.5
L_S = 6
H = 6
L_BP = L_B - L_BC

p = [g, m_B, M_CW, M_P, L_B, H, L_S, L_BC, L_BP]

# Defining equation of motion

def f1(t,z):
    return (z[1],
            -R1fun(z[0],z[1],p)/M1fun(z[0],p))

# Defining event function

def event1(t,z):
    θ = z[0]
    θ_d = z[1]
    θ_dd = -R1fun(z[0],z[1],p)/M1fun(z[0],p)
    φ = np.arcsin(H/L_S - L_BP/L_S*np.sin(z[0] - pi/2))
    
    x_dd = dv_Sfun(θ,θ_d,θ_dd,p)
    
    y_dd = -x_dd * np.tan(φ)
    
    return y_dd - g

def sim1():
    event1.terminal = True
    event1.direction = 0

    # Loading treb data for simulation


    # Defining initial conditions

    z0 = np.array([pi/2 + np.arcsin(H/L_BP),
               0])

    # ODE Solver

    rtol = 1e-6

    sol = solve_ivp(f1, (0,5), z0, rtol = rtol, events = event1)
    t1 = sol.t
    z1 = sol.y
    
    return t1, z1

'''
z1 terms are (θ, θ dot)
'''
