# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

# g       = p[0]
# m_B     = p[1]
# M_CW    = p[2]
# M_P     = p[3]
# L_B     = p[4]
# H       = p[5]
# L_S     = p[6]
# L_BC    = p[7]
# L_BP    = p[8]

# animate_trebuchet
###############################################################################
def animate_trebuchet(t1,z1,t2,z2,t3,z3,p):
    M_P     = p[3]
    L_B     = p[4]
    H       = p[5]
    L_S     = p[6]
    L_BC    = p[7]
    L_BP    = p[8]
    
    FPS = 30    # Target framerate
    
    # Linearise time to target framerate
    t1L = np.linspace(t1[0],t1[-1],round((t1[-1]-t1[0])*FPS+1))
    
    # Relinearise time and displacement interpolate
    thet1f = interp1d(t1, z1[0,:], kind='cubic')
    thet1 = thet1f(t1L)
    phi1 = phifun(thet1,p)
    
    if len(t2) > 0:
        # Linearise time to target framerate
        t2L = np.linspace(t2[0],t2[-1],round((t2[-1]-t2[0])*FPS+1))
        
        # Relinearise time and displacement interpolate
        thet2f = interp1d(t2, z2[0,:], kind='cubic')
        thet2 = thet2f(t2L)
        
        phi2f  = interp1d(t2, z2[1,:], kind='cubic')
        phi2 = phi2f(t2L)
    else:
        t2L = []
        thet2 = []
        phi2 = []
    
    if len(t3) > 0:
        # Linearise time to target framerate
        t3L = np.linspace(t3[0],t3[-1],round((t3[-1]-t3[0])*FPS+1))
        
        # Projectile coordinates (phase 3)
        P_x3f = interp1d(t3, z3[0,:], kind='cubic')
        P_y3f = interp1d(t3, z3[1,:], kind='cubic')
        
        P_x3 = P_x3f(t3L)
        P_y3 = P_y3f(t3L)
    else:
        t3L = []
        P_x3 = []
        P_y3 = []
    
    # Compile 
    t = np.hstack([t1L, t2L, t3L])
    thet = np.hstack([thet1, thet2])
    phi = np.hstack([phi1, phi2])
    
    H2 = -L_BP*np.cos(thet)
    H1 = H - H2
    H3 = -L_BC*np.cos(thet)
    
    d2 = L_BP*np.sin(thet)
    # d1 = np.sqrt( L_S**2 - H1**2 ) - d2
    
    # Beam top coordinates (at centre)
    h_B = 2e-1    # Height of beam
    BT_x = -d2
    BT_y = H1
    
    # Counterweight coordinates
    R_CW = 1                    # Radius of counterweight
    CW_x = L_BC*np.sin(thet)
    CW_y = H + H3
    
    # Beam vertex coordinates
    BV0_x = BT_x + h_B/2*np.cos(thet)
    BV1_x = CW_x + h_B/2*np.cos(thet)
    BV2_x = CW_x - h_B/2*np.cos(thet)
    BV3_x = BT_x - h_B/2*np.cos(thet)
    BV_x = np.vstack((BV0_x.T,
                      BV1_x.T,
                      BV2_x.T,
                      BV3_x.T))
    
    BV0_y = BT_y + h_B/2*np.sin(thet)
    BV1_y = CW_y + h_B/2*np.sin(thet)
    BV2_y = CW_y - h_B/2*np.sin(thet)
    BV3_y = BT_y - h_B/2*np.sin(thet)
    BV_y = np.vstack((BV0_y.T,
                      BV1_y.T,
                      BV2_y.T,
                      BV3_y.T))
    
    beam_verts_END  = np.hstack((BV_x[:,[-1]],BV_y[:,[-1]]))
    
    # Projectile coordinates
    P_x12 = -L_BP*np.sin(thet) - L_S*np.sin(phi - thet)
    P_y12 = H + L_BP*np.cos(thet) - L_S*np.cos(phi - thet)
    
    P_x = np.hstack([P_x12,P_x3])
    P_y = np.hstack([P_y12,P_y3])
    
    rho_water = 1e3                 # Density of water
    V_P = M_P/rho_water             # Volume of projectile
    R_P = np.sqrt(3/4*V_P/np.pi)    # Radius of projectile
    
    # Sling vertex coordinates
    S_w = 5e-2                              # Sling width
    SV0_x = BT_x  + S_w/2*np.cos(phi-thet)
    SV1_x = P_x12 + S_w/2*np.cos(phi-thet)
    SV2_x = P_x12 - S_w/2*np.cos(phi-thet)
    SV3_x = BT_x  - S_w/2*np.cos(phi-thet)
    SV_x = np.vstack((SV0_x.T,
                      SV1_x.T,
                      SV2_x.T,
                      SV3_x.T))
    
    SV0_y = BT_y  - S_w/2*np.sin(phi-thet)
    SV1_y = P_y12 - S_w/2*np.sin(phi-thet)
    SV2_y = P_y12 + S_w/2*np.sin(phi-thet)
    SV3_y = BT_y  + S_w/2*np.sin(phi-thet)
    SV_y = np.vstack((SV0_y.T,
                      SV1_y.T,
                      SV2_y.T,
                      SV3_y.T))
    
    sling_verts_END = np.hstack((SV_x[:,[-1]],SV_y[:,[-1]]))
    
    # Initialise plot
    ###########################################################################
    fig, ax = plt.subplots()
    
    ax.set_title('Press "f" to toggle fullscreen',loc='right')
    
    mngr = plt.get_current_fig_manager()
    mngr.full_screen_toggle()
    
    # Set axis limits and ticks
    XLim = [-(L_B+L_S),(L_B+L_S)]
    
    # Adjust limits to ensure equal
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    YLim = bbox.height/bbox.width*(XLim[1]-XLim[0])
    if YLim < 2*(L_B+L_S):
        YLim = 2*(L_B+L_S)
        dXLim = bbox.width/bbox.height*YLim
        XLim = [-(L_B+L_S),dXLim - (L_B+L_S)]
    
    ax.set_xlim(XLim)
    ax.set_ylim([0,YLim])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Plot projectile path
    plh_PP, = ax.plot(P_x[0], P_y[0], color='blue', linewidth=0.5)
    
    # Plot beam
    beam_verts = np.hstack((BV_x[:,[0]],BV_y[:,[0]]))
    beam_patch = mpatches.Polygon(beam_verts, fc='black')
    ax.add_patch(beam_patch)
    
    # Plot sling
    sling_verts = np.hstack((SV_x[:,[0]],SV_y[:,[0]]))
    sling_patch = mpatches.Polygon(sling_verts, fc='red')
    ax.add_patch(sling_patch)
    
    # Plot counterweight
    CW_patch = mpatches.Circle([CW_x[0],CW_y[0]], radius=R_CW, fc='black')
    ax.add_patch(CW_patch)
    
    # Plot projectile
    P_patch = mpatches.Circle([P_x[0],P_y[0]], radius=R_P, fc='blue')
    ax.add_patch(P_patch)
    
    # Draw frame
    F_w = 1.5e-1 # Frame width
    frame_verts = np.array([[-H/4-F_w,  0], \
                            [-F_w,      H], \
                            [F_w,       H], \
                            [H/4+F_w,   0], \
                            [H/4-F_w,   0], \
                            [0,         H-4*F_w], \
                            [-H/4+F_w,  0] ])
    frame_patch1 = mpatches.Polygon(frame_verts, fc='black')
    frame_patch2 = mpatches.Circle([0,H], radius=F_w, fc='black')
    ax.add_patch(frame_patch1)
    ax.add_patch(frame_patch2)
    
    # Animation function
    ###########################################################################
    def animate(i):
        
        if len(t2) > 0 and t[i] < t2[-1]:
            beam_verts = np.hstack((BV_x[:,[i]],BV_y[:,[i]]))
            sling_verts = np.hstack((SV_x[:,[i]],SV_y[:,[i]]))
            
            beam_patch.set_xy(beam_verts)
            sling_patch.set_xy(sling_verts)
            CW_patch.set_center([CW_x[i],CW_y[i]])
        else:
            beam_patch.set_xy(beam_verts_END)
            sling_patch.set_xy(sling_verts_END)
            CW_patch.set_center([CW_x[-1],CW_y[-1]])
        
        plh_PP.set_data(P_x[0:i+1], P_y[0:i+1])
        P_patch.set_center([P_x[i],P_y[i]])
        
        # Set axis limits
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        dXY = (L_BP+L_S)*0.5    # Border of frame
        
        XLim0 = min(-(L_BP+L_S),min(P_x[0:i+1])) - dXY
        XLim1 = max(+(L_BP+L_S),max(P_x[0:i+1])) + dXY
        dXLim = XLim1 - XLim0
        
        dYLim = max(+(L_BP+L_S),max(P_y[0:i+1])) + dXY
        
        # Adjust limits to ensure equal
        if bbox.width*dYLim > bbox.height*(dXLim):
            dXLim = bbox.width/bbox.height*(dYLim)
            XLim = [(XLim0 + XLim1 - dXLim)/2, (XLim0 + XLim1 + dXLim)/2]
        else:
            XLim = [XLim0, XLim1]
            dYLim = bbox.height/bbox.width*(dXLim)
        
        
        ax.set_xlim(XLim)
        ax.set_ylim([0,dYLim])
        
    # Create animation
    int_time = 1000*(t[-1]-t[0])/(len(t)-1)     # Interval time [ms]
    ani = FuncAnimation(fig = fig,
                        func = animate, 
                        frames = range(len(t)), 
                        interval = int_time, 
                        repeat = False)
    
    return ani



# dphifun
###############################################################################    
def dphifun(thet,dthet,p):
    H       = p[5]
    L_S     = p[6]
    L_BP    = p[8]
    
    dphi = dthet - ( \
                L_BP/L_S*dthet*np.sin(thet)/np.sqrt( 1 - 1/L_S**2*(H+L_BP*np.cos(thet))**2 ) \
           )
    
    return dphi

# dv_Sfun
###############################################################################
def dv_Sfun(thet,dthet,ddthet,p):
    H       = p[5]
    L_S     = p[6]
    L_BP    = p[8]
    
    dv_S = -L_BP**2*dthet**2*np.sin(thet)**2/np.sqrt( L_S**2 - ( H + L_BP*np.cos(thet) )**2 ) \
        - L_BP**2*dthet**2*np.cos(thet)**2/np.sqrt( L_BP**2 - L_BP**2*np.cos(thet)**2 ) \
        + L_BP**2*dthet**2*np.sin(thet)**2/np.sqrt( L_BP**2 - L_BP**2*np.cos(thet)**2 ) \
        + L_BP*ddthet*np.sin(thet)*( H + L_BP*np.cos(thet) )/np.sqrt( L_S**2 - ( H + L_BP*np.cos(thet) )**2 ) \
        + L_BP*dthet**2*np.cos(thet)*( H + L_BP*np.cos(thet) )/np.sqrt( L_S**2 - ( H + L_BP*np.cos(thet))**2 ) \
        - L_BP**2*ddthet*np.cos(thet)*np.sin(thet)/np.sqrt( L_BP**2 - L_BP**2*np.cos(thet)**2 ) \
        - L_BP**2*dthet**2*np.sin(thet)**2*( H + L_BP*np.cos(thet) )**2/( L_S**2 - ( H + L_BP*np.cos(thet))**2 )**(3.0/2.0) \
        + L_BP**4*dthet**2*np.cos(thet)**2*np.sin(thet)**2/( L_BP**2 - L_BP**2*np.cos(thet)**2 )**(3/2)
    return dv_S

# M1fun
###############################################################################
def M1fun(thet,p):
    m_B     = p[1]
    M_CW    = p[2]
    M_P     = p[3]
    L_B     = p[4]
    H       = p[5]
    L_S     = p[6]
    L_BC    = p[7]
    L_BP    = p[8]
    
    M1 = M_P*( 
        L_BP*np.sin(thet)*( H + L_BP*np.cos(thet) )/np.sqrt( -( H + L_BP*np.cos(thet) )**2 + L_S**2 ) \
        - (L_BP/np.sqrt(2)*np.sin(2*thet)/np.sqrt( 1 - np.cos(2*thet) )) \
    )**2 \
    + L_BC**2*M_CW \
    + L_B**3*m_B/3 \
    + L_B*L_BC**2*m_B \
    - L_B**2*L_BC*m_B
    
    return M1

# M2fun
###############################################################################
def M2fun(phi,p):
    m_B     = p[1]
    M_CW    = p[2]
    M_P     = p[3]
    L_B     = p[4]
    L_S     = p[6]
    L_BC    = p[7]
    L_BP    = p[8]
    
    M2_11 = (m_B*L_B**3)/3 - m_B*L_B**2*L_BC + m_B*L_B*L_BC**2 + M_CW*L_BC**2 \
            + M_P*L_BP**2 - 2*M_P*np.cos(phi)*L_BP*L_S + M_P*L_S**2
    M2_12 = -L_S*M_P*( L_S - L_BP*np.cos(phi) )
    M2_22 = L_S**2*M_P
    
    M2 = np.array([[M2_11, M2_12], \
                   [M2_12, M2_22] ])
    return M2

# phifun
###############################################################################
def phifun(thet,p):
    H       = p[5]
    L_S     = p[6]
    L_BP    = p[8]
    phi = thet - np.arccos( ( H + L_BP*np.cos(thet) )/L_S )
    return phi

# R1fun
###############################################################################
def R1fun(thet,dthet,p):
    g       = p[0]
    m_B     = p[1]
    M_CW    = p[2]
    M_P     = p[3]
    L_B     = p[4]
    H       = p[5]
    L_S     = p[6]
    L_BC    = p[7]
    L_BP    = p[8]
    
    R1 = M_P*(
        L_BP*np.sin(thet)*( H + L_BP*np.cos(thet) )/np.sqrt( -( H + L_BP*np.cos(thet) )**2 + L_S**2 ) \
        - np.sqrt(2)*L_BP*np.cos(thet)*np.sin(thet)/np.sqrt( -2*np.cos(thet)**2 + 2 ) \
    )*( \
        L_BP*dthet**2*( np.cos(thet)**2 - np.cos(thet)**4 )/( np.sin(thet)**2 )**(3/2) \
        + ( L_BP*dthet**2*np.sin(thet)**2 )/abs( np.sin(thet) ) \
        - L_BP**2*dthet**2*np.sin(thet)**2/np.sqrt( L_S**2 - ( H + L_BP*np.cos(thet))**2 ) \
        - L_BP*dthet**2*np.cos(thet)**2/np.sqrt( 1 - np.cos(thet)**2 ) \
        + L_BP*dthet**2*np.cos(thet)*( H + L_BP*np.cos(thet) )/np.sqrt( L_S**2 - ( H + L_BP*np.cos(thet) )**2 ) \
        - L_BP**2*dthet**2*np.sin(thet)**2*( H + L_BP*np.cos(thet) )**2/( -( H + L_BP*np.cos(thet) )**2 + L_S**2 )**(3/2) \
    ) + g*np.sin(thet)*( (L_B**2*m_B)/2 + L_BC*M_CW - L_B*L_BC*m_B)
        
    return R1

# R2fun
###############################################################################    
def R2fun(thet,phi,dthet,dphi,p):
    g       = p[0]
    m_B     = p[1]
    M_CW    = p[2]
    M_P     = p[3]
    L_B     = p[4]
    L_S     = p[6]
    L_BC    = p[7]
    L_BP    = p[8]
    
    R2_1 = (g*m_B*np.sin(thet)*L_B**2)/2 - L_BC*g*m_B*np.sin(thet)*L_B \
           - L_BP*L_S*M_P*np.sin(phi)*dphi**2 \
           + 2*L_BP*L_S*M_P*dthet*np.sin(phi)*dphi + L_BC*M_CW*g*np.sin(thet) \
           - L_BP*M_P*g*np.sin(thet) - L_S*M_P*g*np.sin(phi - thet)
    R2_2 = L_S*M_P*( g*np.sin(phi - thet) - L_BP*np.sin(phi)*dthet**2 )
    
    return [R2_1, R2_2]

# load_treb_data
###############################################################################
def load_treb_data(L_BC,L_S,H):
    fname = "data_DO_NOT_RENAME/treb_data_" + str(round(L_BC*10)) + ".npy"
    data = np.load(fname,allow_pickle=True)

    L_SA = data[1]
    HA = data[2]
    t1A = data[3]
    t2A = data[4]
    t3A = data[5]
    z1A = data[6]
    z2A = data[7]
    z3A = data[8]

    idx = (abs(L_SA-L_S)<0.05) & (abs(HA-H)<0.05)
    
    t1 = t1A[idx][0]
    t2 = t2A[idx][0]
    t3 = t3A[idx][0]
    z1 = z1A[idx][0]
    z2 = z2A[idx][0]
    z3 = z3A[idx][0]
    
    return t1, z1, t2, z2, t3, z3