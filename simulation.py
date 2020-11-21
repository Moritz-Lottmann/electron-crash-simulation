import numpy as np
from numpy import cos, sin, arctan
from scipy.constants.constants import e, pi, epsilon_0, m_e, c
import matplotlib.pyplot as plt
dt = 1e-23
n = int(2e7)

li_r = []

r_l0 = 0.52917721e-10
a_l0 = e ** 2 / (4 * pi * epsilon_0 * r_l0 ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
v_l0 = (a_l0*r_l0)**(1/2)

r = r_l0*np.array([0, 1])
r_l = np.linalg.norm(r)
r_e = r/r_l
v = v_l0*np.array([-1, 0.])
a = a_l0*(-r_e)

li = []
li_v_l = []
li_dv = []

for i in range(n):
    # plotting & tracking
    if i % (n/10) == 0:
        print(round((i/n)*100, 2), '%')
    if i % (n/1000) == 0:
        li.append([r[0],r[1]])

    # calculations newton
    r += v*dt
    v += a*dt
    v_l = np.linalg.norm(v)
    r_l = np.linalg.norm(r)
    r_e = r / r_l
    v_e = v / v_l
    a_l = e ** 2 / (4 * pi * epsilon_0 * r_l ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
    a = a_l*(-r_e)

    # # calculation radiation
    # en_0 = 1/2*m_e*v_l**2
    # p = e**2/6/pi/epsilon_0/c**3*abs(a_l)**2
    # en_delta = p*dt
    #
    # en_new = en_0-en_delta
    # v_l = (2*en_new/m_e)**(1/2)
    # v = v_e * v_l
    li_v_l.append(v_l)


li_np = np.array(li)
li_np = np.transpose(li_np)
plt.plot(li_np[0], li_np[1])
# plt.plot(li_v_l)
plt.axes().set_aspect('equal', 'datalim')
plt.show()
