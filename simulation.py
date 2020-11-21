import numpy as np
from numpy import cos, sin, arctan
from scipy.constants.constants import e, pi, epsilon_0, m_e
import matplotlib.pyplot as plt
dt = 0.002
n = 100000

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
    r += v*dt
    if i % 200 == 0:
        plt.plot(r[0],r[1], marker="x")
    v += a*dt
    r_l = np.linalg.norm(r)
    r_e = r / r_l
    a_l = e ** 2 / (4 * pi * epsilon_0 * r_l ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
    a = a_l*(-r_e) / r_l ** 2

# print(li_r)
plt.axes().set_aspect('equal', 'datalim')
plt.show()