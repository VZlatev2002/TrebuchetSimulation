import numpy as np
from scipy.integrate import solve_ivp
import Trebuchet_Functions as tf
from matplotlib.animation import FuncAnimation
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
import math

# System Parameters

g = 9.81
m_B = 12
M_CW = 5000
M_P = 250
L_B = 10 #10
p_a = 1.225 # density of air
p_w = 1000 # density of water
C_d = 0.5 # drag coefficient
pi = np.pi
L_BC = 5.0#distance between the pivot and counterweight
L_S = 8.0 # the sling length 
H = 4.0 # height
L_BP = L_B - L_BC #the difference

p = [g, m_B, M_CW, M_P, L_B, H, L_S, L_BC, L_BP]


def f1(t,z):
    return (z[1],
            -tf.R1fun(z[0],z[1],p)/tf.M1fun(z[0],p))

# Defining event function

def event1(t,z):
    θ = z[0]
    θ_d = z[1]
    θ_dd = -tf.R1fun(z[0],z[1],p)/tf.M1fun(z[0],p)
    φ = np.arcsin(H/L_S - L_BP/L_S*np.sin(z[0] - pi/2))
    
    x_dd = tf.dv_Sfun(θ,θ_d,θ_dd,p)
    
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

def f2(t, z):
    
    M2 = tf.M2fun(z[1],p)
    R2 = tf.R2fun(z[0],z[1],z[2],z[3],p)
    
    x, y = symbols('x y')
    eq1 = Eq(M2[0][0]*x + M2[0][1]*y + R2[0], 0)
    eq2 = Eq(M2[1][0]*x + M2[1][1]*y + R2[1], 0)
    
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

t1, z1 = sim1()
t2, z2 = sim2(t1, z1)
t3, z3 = sim3(t2, z2)
print(z3[0][-1])
ani = tf.animate_trebuchet(t1, z1, t2, z2, t3, z3, p)
             
# 3,6,6.5 -> 153.355
#ani = tf.animate_trebuchet(t1, z1, t2, z2, t3, z3, p)