#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 02:05:06 2022

@author: tolgakaancelebi
"""

import numpy as np
from scipy.integrate import solve_ivp

from Trebuchet_Functions import load_treb_data, animate_trebuchet

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

# Defining drag function

def D_p(v):
    return 0.5 * C_d * p_a * pi * ((3 * M_P)/(4 * pi * p_w)) ** (2/3) * v / np.linalg.norm(v)

# Defining equation of motion

def f3(t,z):
    
    v = np.array([z[2],z[3]])
    
    return (z[2],
            z[3],
            -D_p(v)[0] / M_P * z[2]**2,
            -D_p(v)[1] / M_P * z[3]**2 - g)

# Defining event function

def event3(t,z):
    return z[1]




def sim3(t2, z2):
    event3.terminal = True
    event3.direction = -1
# Defining initial conditions based on data from phase 2

    z00 = -(L_BP+L_S)*np.sin(z2[0][-1])
    z01 = H+(L_BP+L_S)*np.cos(z2[0][-1])
    z02 = (L_S*np.cos(z2[0][-1]-z2[1][-1])-(L_BP+L_S)*np.cos(z2[0][-1]))*z2[2][-1]\
        - L_S*np.cos(z2[0][-1]-z2[1][-1])*z2[3][-1]
    z03 = (L_S*np.sin(z2[0][-1]-z2[1][-1])-(L_BP+L_S)*np.sin(z2[0][-1]))*z2[2][-1]\
        - L_S*np.sin(z2[0][-1]-z2[1][-1])*z2[3][-1]
    
    z0 = np.array([z00,
          z01,
          z02,
          z03])
    
    # ODE Solver
    
    rtol = 1e-6
    
    sol = solve_ivp(f3, (t2[-1],t2[-1]+20), z0, rtol = rtol, events = event3)
    t3 = sol.t
    z3 = sol.y
    
    return t3, z3
