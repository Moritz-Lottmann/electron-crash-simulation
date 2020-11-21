import numpy as np
from numpy import cos, sin, arctan
from scipy.constants.constants import e, pi, epsilon_0, m_e, c
import matplotlib.pyplot as plt
dt = 1e-21
n = 2000000

li_r = []

r_l0 = 0.52917721e-10
a_l0 = e ** 2 / (4 * pi * epsilon_0 * r_l0 ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
v_l0 = (a_l0*r_l0)**(1/2)

r = r_l0*np.array([0, 1])
r_l = np.linalg.norm(r)
r_e = r/r_l
v = v_l0*np.array([-1, 0.])
a = a_l0*(-r_e)


for i in range(n):
    # plotting & tracking
    if i % (n/100) == 0:
        plt.plot(r[0],r[1], marker="x")
        li_r.append([r[0],r[1]])

    # calculations newton
    r += v*dt
    v += a*dt
    v_l = np.linalg.norm(v)
    r_l = np.linalg.norm(r)
    r_e = r / r_l
    v_e = v / v_l
    a_l = e ** 2 / (4 * pi * epsilon_0 * r_l ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
    a = a_l*(-r_e)

    # calculation radiation
    en_0 = 1/2*m_e*v_l**2
    p = e**2/6/pi/epsilon_0/c**3*abs(a_l)**2
    en_delta = p*dt
    en_new = en_0-en_delta
    v_l = (2*en_new/m_e)**(1/2)
    v = v_e * v_l


# print(li_r)
plt.axes().set_aspect('equal', 'datalim')
plt.show()
