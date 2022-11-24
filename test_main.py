import numpy as np
from scipy.integrate import solve_ivp
import Trebuchet_Functions as tf
from matplotlib.animation import FuncAnimation
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
import math
import time
from joblib import Parallel, delayed

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
L_BC = 1.0 #distance between the pivot and counterweight
L_S = 5.0 # the sling length 
H = 5.0 # height
L_BP = L_B - L_BC #8.5

p = [g, m_B, M_CW, M_P, L_B, H, L_S, L_BC, L_BP]


def f1(t,z):
    return (z[1],
            -tf.R1fun(z[0],z[1],p)/tf.M1fun(z[0],p))

# Defining event function

def event1(t,z):
    θ = z[0]
    θ_d = z[1]
    θ_dd = -tf.R1fun(z[0],z[1],p)/tf.M1fun(z[0],p)
    φ = np.arcsin(H/L_S - L_BP/L_S*np.sin(z[0] - pi/2)) # Use this for the first constrain instrad of H/L_BP
    
    x_dd = tf.dv_Sfun(θ,θ_d,θ_dd,p)
    
    y_dd = -x_dd * np.tan(φ)
    
    return y_dd - g

event1.terminal = True
event1.direction = 0

def sim1():
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

event2.terminal = True
event2.direction = 1


#ODE Solver & initial conditions
def sim2(t1, z1):
    phi = tf.phifun(z1[0][-1],p)
    dphi = tf.dphifun(z1[0][-1], z1[1][-1],p)

    # Equation of motion

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



L_BCa = np.arange(3.0, 4.1, 0.5)
L_Sa = np.arange(5.0, 7.1, 0.5)
Ha = np.arange(5.0, 7.1, 0.5)
    
def main(i,j,k):
    global L_BC
    global L_S
    global H
    global p
    global L_BP
    L_BC, L_S, H = i,j,k
    L_BP = L_B - L_BC
    p = [g, m_B, M_CW, M_P, L_B, H, L_S, L_BC, L_BP]
    if ( H/L_BP <= -1 or H/L_BP >= 1): # arcsin  constrains (not a real physical constrain, more mathematical)
        return (0, 0)
    elif (M_CW*L_BC < L_BP*M_P): # Moments relationship
        return (0, 0)
    else:
        t1, z1 = sim1()
        t2, z2 = sim2(t1, z1)
        if (np.cos(z2[0][-1])*(L_BC+1) > H):
            return (0, 0)
        elif (z2[0][-1] < 70/180 * np.pi ): # If theta is below 70 degrees, then the distance is going to be negative; this more optimization rather than physical constrain
            return (0,0)
        t3, z3 = sim3(t2, z2)
        point = "{},{},{}".format(i,j,k)
        
        print(round(z3[0][-1], 3), point)
        #ani = tf.animate_trebuchet(t1, z1, t2, z2, t3, z3, p)
        #plt.show()
        
    return (round(z3[0][-1], 3), point)
   
    # ani = tf.animate_trebuchet(t1, z1, t2, z2, t3, z3, p)      
t1 = time.time()
x_y = Parallel(n_jobs=7)(delayed(main)(i,j,k) for i in L_BCa for j in L_Sa for k in Ha)
np.save('simulation_data.npy',x_y)
print(x_y)
distance_x = [i[0] for i in x_y]
coordinate = [i[1] for i in x_y]
#ani = tf.animate_trebuchet(t1, z1, t2, z2, t3, z3, p)

# print(distance_x)
# print(coordinate)
max_value = max(distance_x)
max_index = distance_x.index(max_value)
print(max_value)
coordinates = coordinate[max_index]
print(coordinates)



#fig,ax1 = plt.subplots()
#ax1.plot(coordinate, distance_x, '.-')
#plt.show()
