#!/usr/bin/python -tt
""" Simulate a distillation """
import sys
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

eps = sys.float_info.epsilon

def Antoine(T, species):
    """ Find individual vapor pressure as a function of temperature.
        Each species has characteristic constants. """
    antoine_dict = {'water':[5.0768, 1659.793, -45.854],
                    'ethanol':[5.24677, 1598.673, -46.424],
                    'propanoic acid':[4.74558, 1679.869, -59.832],
                    'butanoic acid':[4.90904, 1793.898, -70.564],
                    'pentanoic acid':[3.2075, 879.771, -172.237],
                    'hexanoic acid':[4.34853, 1512.718, -129.255],
                    'heptanoic acid':[4.30691, 1536.114, -137.446],
                    'octanoic acid':[4.25235, 1530.446, -150.12],
                    'nonanoic acid':[2.54659, 733.594, -235.239],
                    'decanoic acid':[2.4645, 733.581, -256.708],
                    'isopropanol':[4.8610, 1357.427, -75.814],
                    '1-butanol':[4.50393, 1313.878, -98.789],
                    'isobutanol':[4.43126, 1236.911, -101.528],
                    '2-pentanol':[4.42349, 1291.212, -100.017],
                    '1-pentanol':[4.68277, 1492.549, -91.621],
                    'hexanol':[4.41271, 1422.031, -107.706],
                    'heptanol':[3.9794, 1256.783, -133.487],
                    'octanol':[3.74844, 1196.639, -149.043],
                    'nonanol':[3.96157, 1373.417, -139.182],
                    'decanol':[3.51869, 1180.306, -168.829]}
    A, B, C = antoine_dict[species]
    return 10**(A-B/(C+T)) # bar

def emperical_vapor_composition(abv):
    """ Empirical formula for vapor composition of a binary ethanol:water mixture."""
    #print('abv = %1.2f' %abv)
    vapor_abv = -94.7613*abv**8.0 + 450.932*abv**7.0 -901.175*abv**6.0 + 985.803*abv**5.0 - 644.997*abv**4.0 + 259.985*abv**3.0 - 64.5050*abv**2.0 + 9.71706*abv
    #print('vapor_abv = %1.2f' %vapor_abv)
    return vapor_abv

def calculated_vapor_composition(abv): # Doesn't take into account deviations from Raoult
    """ Vapor composition calculated from Antoine and Raoult """
    T = temp(abv)

    ethanol_vapor_pressure = Antoine(T,'ethanol') # bar
    water_vapor_pressure = Antoine(T,'water') # bar

    ethanol_density = 0.789 # g/mL
    ethanol_MW = 46.06844 # g/mol

    water_density = 1.0 # g/mL
    water_MW = 18.01528 # g/mol

    ethanol_molar_fraction = (abv*ethanol_density/ethanol_MW) / (abv*ethanol_density/ethanol_MW + (1-abv)*water_density/water_MW)
    water_molar_fraction = ((1-abv)*water_density/water_MW) / (abv*ethanol_density/ethanol_MW + (1-abv)*water_density/water_MW)

    ethanol_partial_pressure = ethanol_vapor_pressure*ethanol_molar_fraction # bar
    water_partial_pressure = water_vapor_pressure*water_molar_fraction # bar
    return ethanol_partial_pressure / (ethanol_partial_pressure + water_partial_pressure)

def temp(abv):
    """ Calculate temperature for a boiling ethanol:water binary mixture """
    if abv < eps:
        T = 300.0
    elif 1.0 - abv < eps:
        T = 300.0
    else:
        T = 60.526*abv**4.0 - 163.16*abv**3.0 + 163.96*abv**2.0 - 83.438*abv + 100.0 + 273.15 # K
    #print T
    return T

def vaporization_enthalpy(T,species):
    enthalpy_dict = {'ethanol':[50.43, -0.4475, 0.4989, 413.9]}
    A,alpha,beta,Tc = enthalpy_dict[species]
    return A*np.e**(-alpha*(T/Tc))*(1.0-(T/Tc))**beta*1000.0 # J/mol

def binary_vaporization_rate(abv,power,T):
    """ Assume the vaporization rate of the mix is a molar combination of individual vaporization rates """
    ethanol_density = 0.789 # g/mL
    ethanol_MW = 46.06844 # g/mol
    #ethanol_enthalpy = vaporization_enthalpy(T,'ethanol') # J/mol
    ethanol_enthalpy = 38600.0 # J/mol
    water_density = 1.0 # g/mL
    water_MW = 18.01528 # g/mol
    water_enthalpy = 40650.0 # J/mol
    #abv = 0.1
    vap_rate = power/(abv*ethanol_density/ethanol_MW*ethanol_enthalpy + (1.0-abv)*water_density/water_MW*water_enthalpy) # mL of liquid / s
    #print('vaporization rate = %5.5f' %vap_rate)
    return vap_rate

def reflux(ethanol,water):
    """ Return reflux rate as a function of liquid height in a plate """
    plate_hole_radius = 0.0005 # m
    r = .0779/2.0 # column radius in m
    C = ((2*9.81)**(0.5))*(np.pi*plate_hole_radius**2.0) # C has units of m(3/2)/s. This constant can be tuned, though sqrt(2g)*area is a good place to start
    if (ethanol + water) < eps:
        height = eps
    else:
        height = ((ethanol+water)/1000000.0)/(np.pi*r**2.0) # ethanol and water in are in mL, height is in mL
    reflux = 1000000.0*C*height**(0.5) # mL/s
    #print 'C = ' + str(C)
    #print 'height = ' + str(height)
    #print 'reflux = ' + str(reflux)
    return reflux
def congener_mole_fraction_float(congener,ethanol,water):

    ethanol_density = 0.789 # g/mL
    ethanol_MW = 46.06844 # g/mol

    water_density = 1.0 # g/mL
    water_MW = 18.01528 # g/mol

    if (congener + ethanol*ethanol_density/ethanol_MW + water*water_density/water_MW) < eps:
        congener_mole_fraction = 0.0
    else:
        congener_mole_fraction = congener / (congener + ethanol*ethanol_density/ethanol_MW + water*water_density/water_MW)

    return congener_mole_fraction

def congener_molarity_float(congener,ethanol,water):
    if (ethanol + water) < eps:
        congener_molarity = 0.0 # Could change this to the actual molarity of the pure congener?
    else:
        congener_molarity = congener/(ethanol + water)
    return congener_molarity

def abv_float(ethanol,water):
    if (ethanol < eps) | (water < eps):
        abv = eps
    elif (ethanol + water) < eps:
        abv = eps
    else:
        abv = ethanol/(ethanol + water)
    return abv

def initialize(ethanol,water,plates,congener):
    col0 = []
    for i in range(0,(plates*3),3):
        #col0.append(ethanol/1000)
        #col0.append(water/1000)
        #col0.append(congener/1000)
        col0.append(0.0)
        col0.append(0.0)
        col0.append(0.0)
    col0[0:3] = [ethanol, water, congener] # initialize the pot
    return col0

def col(y,t,plates,congener_identity,reflux_knob,power):
    """
    Col is a vector of the form [e1, h1, c1, e2, h2, c2...en, hn, cn] where e and h are mL, and cn is molarity of a congener (takes up 0 volume)
    """

    #ethanol_tot =
    #water_tot =
    #congener_tot =

    """
    for i in range(len(y)):
        if y[i] < eps:
            y[i] = 0.0
    """

    ethanol_density = 0.789 # g/mL
    ethanol_MW = 46.06844 # g/mol

    water_density = 1.0 # g/mL
    water_MW = 18.01528 # g/mol

    tube_area = np.pi*(.007899/2.0)**2 # m^2, cross sectional area of liquid management tubing
    Vdead = 0.330*tube_area # m^3 This volume must be occupied before fluid can go to reflux 330 mm is just an estimate
    h2max = 0.011 # m, a design parameter, could also measure actual h2
    Vtakeoff = Vdead + (h2max*tube_area)*2.0 # m^3 Once this volume is occupied, all excess goes to takeoff
    #print('Vdead = %2.10f' %Vdead) #16.17 mL
    #print('Vtakeoff = %2.10f' %Vtakeoff) #17.25 mL

    dydt = []
    for i in range(plates*3):
        dydt.append(0)

    for i in range(0,(len(dydt)-3),3):

        if y[i] > eps:
            ethanol = y[i] # mL
        else:
            ethanol = 0.001
        if y[i+1] > eps:
            water = y[i+1] # mL
        else:
            water = 0.001

        abv = abv_float(ethanol,water)
        vapor_abv = emperical_vapor_composition(abv)
        T = temp(abv) # K
        #vapor_abv = calculated_vapor_composition(abv)
        reflux_rate = reflux(ethanol,water) # mL/s
        vap_rate = binary_vaporization_rate(abv,power,T) # mL of liquid / s

        congener = y[i+2] # mol
        congener_molarity = congener_molarity_float(congener,ethanol,water)
        congener_mole_fraction = congener_mole_fraction_float(congener,ethanol,water)
        congener_vapor_pressure = Antoine(T,congener_identity) # bar
        congener_partial_pressure = congener_vapor_pressure*congener_mole_fraction # bar

        #print 'i = ' + str(i) + ' water = ' + str(water) + ' ethanol = ' + str(ethanol) + ' abv = ' + str(abv) + ' vapor rate ' + str(vap_rate) + ' t = ' + str(t)
        if i == 0: # Pot

            if y[i] > eps:
                dydt[i] += - vapor_abv*vap_rate # current plate ethanol
            if y[i+1] > eps:
                dydt[i+1] += -(1.0-vapor_abv)*vap_rate # current plate water
            if y[i+2] > eps:
                dydt[i+2] += - congener_molarity*congener_partial_pressure/1.01325*vap_rate# current plate congener

            if y[i] > eps:
                dydt[i+3] += vapor_abv*vap_rate # upper plate ethanol
            if y[i+1] > eps:
                dydt[i+4] += (1.0-vapor_abv)*vap_rate # upper plate water
            if y[i+2] > eps:
                dydt[i+5] += congener_molarity*congener_partial_pressure/1.01325*vap_rate # upper plate congener

            #print abv
        elif i == (len(y)-6): # Reverse Liquid Management Head

            if ((ethanol + water)/1000000 - Vdead) > eps:
                h2 = ((ethanol+water)/1000000 - Vdead)/(2*tube_area) # m, height driving flow through needle valve
            else:
                h2 = eps
            if h2 < eps:
                h2 = eps
            """
            if (h2 - h2max) > eps:
                takeoff = (2*9.81*(h2-h2max))**(0.5)*tube_area*1000000
                #h2_reflux = h2max
            elif h2 < eps:
                h2 = eps
                h2reflux = eps
                takeoff = eps # formerly not here
            else:
                takeoff = eps # formerly not here
            #print h2
            if takeoff < eps:
                takeoff = eps
            """

            Cv = 0.43*reflux_knob # flow coefficient
            Kv = 0.865*Cv # flow factor (metric)
            #rho = (0.5*ethanol_density + 0.5*water_density)
            rho = abv*ethanol_density + (1.0-abv)*water_density # g/mL Including abv here creates instability.
            if rho < eps:
                rho = eps
            elif 1-rho < eps:
                rho = 1-eps
            SG = rho/water_density
            DP = h2*rho*1000.0*9.81 # change in pressure, Pa = m*kg/m^3 *m/s^2 = kg*m/s^2 * 1/m^2 = N/m^2
            Q = Kv*((DP/100000.0)/SG)**(0.5) # m^3/h
            head_reflux = Q*1000000.0/3600.0 # mL/s
            if head_reflux < eps:
                head_reflux = eps
            #head_reflux = 0.2
            #takeoff = 0.0
            #if h2 > h2max:
            #    takeoff = vap_rate-head_reflux
            if ((h2-h2max) > eps) & ((vap_rate - head_reflux) > eps):
                takeoff = vap_rate - head_reflux
            else:
                takeoff = 0.0

            if takeoff < eps:
                takeoff = eps
            #takeoff = 0.0
            #head_reflux = 0.0
            #print('takeoff = %2.2f' %takeoff)
            #eps1 = 0.0000000000000001
            if y[i] > eps:
                dydt[i-3] += abv*head_reflux # lower plate ethanol
            if y[i+1] > eps:
                dydt[i-2] += (1.0-abv)*head_reflux # lower plate water
            if y[i+2] > eps:
                dydt[i-1] += congener_molarity*head_reflux # lower plate congener

            if y[i] > eps:
                dydt[i] += -abv*head_reflux - abv*takeoff # current plate ethanol
            if y[i+1] > eps:
                dydt[i+1] += -(1.0-abv)*head_reflux - (1.0-abv)*takeoff # current plate water
            if y[i+2] > eps:
                dydt[i+2] += -congener_molarity*head_reflux - congener_molarity*takeoff # current plate congener

            if y[i] > eps:
                dydt[i+3] += abv*takeoff  # takeoff  ethanol
            if y[i+1] > eps:
                dydt[i+4] += (1.0-abv)*takeoff # takeoff plate water
            if y[i+2] > eps:
                dydt[i+5] += congener_molarity*takeoff # takeoff plate congener
            #print abv
            """
            else if i = len(y) - 6: # liquid management head





            """
        else: # any other plate

            if y[i] > eps:
                dydt[i-3] += abv*reflux_rate # lower plate ethanol
            if y[i+1] > eps:
                dydt[i-2] += (1.0-abv)*reflux_rate # lower plate water
            if y[i+2] > eps:
                dydt[i-1] += congener_molarity*reflux_rate # lower plate congener

            if y[i] > eps:
                dydt[i] += -abv*reflux_rate - vapor_abv*vap_rate # current plate ethanol
            if y[i+1] > eps:
                dydt[i+1] += -(1.0-abv)*reflux_rate -(1.0-vapor_abv)*vap_rate # current plate water
            if y[i+2] > eps:
                dydt[i+2] += -congener_molarity*reflux_rate - congener_molarity*congener_partial_pressure/1.01325*vap_rate # current plate congener

            if y[i] > eps:
                dydt[i+3] += vapor_abv*vap_rate # upper plate ethanol
            if y[i+1] > eps:
                dydt[i+4] += (1.0-vapor_abv)*vap_rate # upper plate water
            if y[i+2] > eps:
                dydt[i+5] += congener_molarity*congener_partial_pressure/1.01325*vap_rate # upper plate congener

    #print dydt
    return dydt

def distill(ethanol,water,plates,congener,congener_identity,reflux_knob,power,timestep,tend):
    """
    Units:
        ethanol - initial ethanol in mL
        water - initial water in mL
        plates - 1 for the pot, 1 for the head, 1 for the takeoff
        congener - initial congener in mol
        congener_identity - text
        reflux_knob -- setting on the reflux valve, 0-1
        power - W
        timestep- s
        time - s
    """

    col0 = initialize(ethanol,water,plates,congener)
    t = np.linspace(0,tend,timestep)
    sol = odeint(col, col0, t, args=(plates,congener_identity,reflux_knob,power),full_output=0,rtol=1.0,atol=1.0)
    #print sol
    """
    plt.figure(1)
    plt.subplot(511)
    plt.plot(t,sol[:,0], 'b', label='ethanol|pot(t)')
    plt.plot(t,sol[:,1], 'g', label='water|pot(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(512)
    plt.plot(t,sol[:,3], 'b', label='ethanol|plate1(t)')
    plt.plot(t,sol[:,4], 'g', label='water|plate1(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(513)
    plt.plot(t,sol[:,6], 'b', label='ethanol|plate2(t)')
    plt.plot(t,sol[:,7], 'g', label='water|plate2(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(514)
    plt.plot(t,sol[:,9], 'b', label='ethanol|head')
    plt.plot(t,sol[:,10], 'g', label='water|head')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(515)
    plt.plot(t,sol[:,12], 'b', label='ethanol|takeoff')
    plt.plot(t,sol[:,13], 'g', label='water|takeoff')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
    """

    plt.figure(1)

    plt.subplot(10,1,1)
    plt.plot(t,sol[:,0]/(sol[:,0] + sol[:,1]), 'b', label='abv|pot')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,2)
    plt.plot(t,(sol[:,0] + sol[:,1]), 'g', label='Volume (mL)|pot')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,3)
    plt.plot(t,sol[:,3]/(sol[:,3] + sol[:,4]), 'b', label='abv|plate 1')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,4)
    plt.plot(t,(sol[:,3] + sol[:,4]), 'g', label='Volume (mL)|plate 1')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,5)
    plt.plot(t,sol[:,6]/(sol[:,6] + sol[:,7]), 'b', label='abv|plate 2')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,6)
    plt.plot(t,(sol[:,6] + sol[:,7]), 'g', label='Volume (mL)|plate 2')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,7)
    plt.plot(t,sol[:,9]/(sol[:,9] + sol[:,10]), 'b', label='abv|head')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,8)
    plt.plot(t,(sol[:,9] + sol[:,10]), 'g', label='Volume (mL)|head')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,9)
    plt.plot(t,sol[:,12]/(sol[:,12] + sol[:,13]), 'b', label='abv|takeoff')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.subplot(10,1,10)
    plt.plot(t,(sol[:,12] + sol[:,13]), 'g', label='Volume (mL)|takeoff')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()


    plt.show()


if __name__ == "__main__":
    distill(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]))

    #distill(ethanol,water,plates,congener,congener_identity,reflux_knob,power,timestep,tend)
