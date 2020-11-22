import numpy as np
from numpy import cos, sin, arctan
from scipy.constants.constants import e, pi, epsilon_0, m_e, c
import matplotlib.pyplot as plt
k = 1
r_l = 1e-10#0.52917721e-10
a_0 = e ** 2 / (4 * pi * epsilon_0 * r_l ** 2 * m_e) # Beschleunigung aufgrund der Coulombkraft
v_l = (a_0*r_l)**(1/2)

li_r = []

elapsed_time = 0
finished = False
n = 0
while not finished:
    # plotting & tracking
    if n % 100000 == 0:
        print(int(n/1000), "k Rechnungen")
    n += 1

    if r_l > 1e-15:
        # calculate T
        T = 2 * pi * r_l / v_l / k
        elapsed_time += T

        # calculation radiation
        en_0 = 1/2*m_e*v_l**2
        a_l = v_l**2/r_l
        p = e**2/6/pi/epsilon_0/c**3*abs(a_l)**2
        en_delta = p*T
        en_new = en_0-en_delta

        v_l = (2*en_new/m_e)**(1/2)
        r_l = v_l**2/a_l
        if n % 10000 == 0:
            li_r.append([r_l, elapsed_time])
    else:
        finished = True

li_r_np = np.array(li_r)
li_r_np = np.transpose(li_r_np)
li_dr = []
for i in range(len(li_r_np[0])-1):
    li_dr.append((li_r_np[0][i]-li_r_np[0][i+1])/li_r_np[0][i]*100)
li_dr_np = np.array(li_dr)
plt.plot(li_r_np[0], marker="x")#, li_r_np[0], marker="x")
print("Total n/k:", n/k)
print("Elapsed time: ", elapsed_time)
#plt.axes().set_aspect('equal', 'datalim')
plt.show()
