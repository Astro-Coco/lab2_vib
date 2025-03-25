import scipy.integrate as integrate
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import cmath
import sys

# # Use LaTeX fonts in the plot
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# Données
Mav = 47500         # masse de l'avion au complet
Mfus = 46266.1      # masse de l'avion sans landing gear
Mnlg = 145.3        # masse NLG
Mmlg = 2*544.3      # masse des MLG
Ifus = 1.6961e6     # inertie de l'avion sans train
Dnm = 14.441        # distance entre nose et main
d2 = 13.618         # distance entre nose et cg
d1 = Dnm-d2         # distance entre main et cg
Ampl = 0.01         # amplitude de la route
phi = -2*np.pi*0.441    # déphasage entre nose et main
V = 7.71667         # vitesse de taxi (m/s)
omega = 2*np.pi*V   # fréquence déterminée à partir de la vitesse de taxi et la forme de la route
# omega = 111.19499981
# omega = 110.14156318
# omega = 2.31928416
# omega = 8.56023595
print("Omega", omega)
CA1 = 2*13328.
CA2 = 2962. 
KA1 = 2*2895315.
KA2 = 46266.1
Km = 2*3891698.
Kn = 1732578.
Nddl = 4

# QUESTION 1
# On forme les matrices
M = np.array(
    [[Mmlg, 0, 0, 0],
     [0, Mnlg, 0, 0],
     [0, 0, Mfus, 0],
     [0, 0, 0, Ifus]])

print("M : ", M)
invM = np.linalg.inv(M)
print("invM : ",invM)

C = np.array(
    [[CA1,0,-CA1,d1*CA1],
     [0,CA2,-CA2,-d2*CA2],        
     [-CA1,-CA2, CA1+CA2,d2*CA2-d1*CA1],
     [d1*CA1,-d2*CA2,d2*CA2-d1*CA1,d1**2*CA1+d2**2*CA2]])

print("C : ",C)

K = np.array(
    [[KA1+Km,0,-KA1,d1*KA1],
     [0,KA2+Kn,-KA2,-d2*KA2],
     [-KA1,-KA2,KA1+KA2,d2*KA2-d1*KA1],
     [d1*KA1,-d2*KA2,d2*KA2-d1*KA1,d2**2*KA2+d1**2*KA1]])

print("K : ",K)

F = np.array([Ampl*Km, Ampl*Kn, 0, 0])
Phase = np.array([phi,0,0,0])

# On calcule phi et lambda et on en déduit tout le reste
LAMBDA, PHI = la.eig(K,M)
print("LAMBDA : Les valeurs propres")
print(LAMBDA)
print("PHI normalisée (norme euclidienne)")
print(PHI)
Wi = np.sqrt(np.real(LAMBDA))
print("Les quatre fréquences propres en rad/s sont")
print(Wi)

print("Matrice masse modale")
Mm = np.diag(np.diag(PHI.transpose()@M@PHI))
PHI=PHI@np.sqrt(np.linalg.inv(Mm))
print("PHI telle que Phit.M.Phi=I")
print(PHI)
Mm = PHI.transpose()@M@PHI
print(Mm)

print("Matrice amortissement modale")
Cm = PHI.transpose()@C@PHI
print("Cm", Cm)
print("ratios d'amortissement")
zeta = np.diag(Cm)/(2*Wi)
print(zeta)

Fm = PHI.transpose()*F

# number of time points
n = 40001

# time points
t = np.linspace(0, 100, n)

# initial condition
z0 = np.zeros(2*Nddl)
# z0 = [0., 0., 0., 0., 0., 0.4, 0., 0.1]
# storage
x = np.zeros((n, Nddl))
y = np.zeros((n, Nddl))
# Forcing term
u = np.zeros((n, Nddl))
for i in range(n):
    u[i][:]=F[:]*np.cos(omega*t[i]+Phase[:])

# record initial conditions
for i in range(Nddl):
    x[0][i] = z0[2*i]
    y[0][i] = z0[2*i+1]

# function that returns dz/dt
def model(z,t,u,invM,c,k,n):
    xl = np.zeros(n)
    yl = np.zeros(n)
    dzdt = np.zeros(2*n)
    for i in range(n):
        xl[i] = z[2*i]
        yl[i] = z[2*i+1]
    dxdt = yl
    dydt = invM @ ( -c @ yl - k @ xl + u)
    for i in range(n):
        dzdt[2*i] = dxdt[i]
        dzdt[2*i+1] = dydt[i]
    return dzdt

# solve ODE
for i in range(1, n):
    # span for next time step
    tspan = [t[i-1], t[i]]
    # solve for next step
    z = odeint(model, z0, tspan, args=(u[i][:], invM, C, K, Nddl))
    
    if z.shape[1] == 2:  # Si la sortie est de forme (2, 2), ajuste les indices
        for j in range(Nddl):
            x[i, j] = z[-1, 0]  # Prend la dernière valeur de la première colonne
            y[i, j] = z[-1, 1]  # Prend la dernière valeur de la deuxième colonne
    else:
        for j in range(Nddl):
            x[i, j] = z[-1, 2*j]   # Correctement dimensionné
            y[i, j] = z[-1, 2*j+1]

    # next initial condition
    z0[:] = z[-1, :]


# Seulement en mode forçage. À inclure sur les figures de fonction de transfert
Amax = np.zeros(Nddl)
for i in range(Nddl):
    print("Valeur max. en régime permanent : ", np.amax(x[int(n/2):,i]))
    Amax[i] = np.amax(x[int(n/2):,i])

# QUESTION 1 f)
# On calcule les fonctions de transfert d'amplitude
nsamples = 1000
w = np.linspace(0, 120, nsamples)
A = np.zeros((nsamples,Nddl))
Dephasage = np.zeros((nsamples,Nddl))
for i in range(nsamples):
    A[i,:]=abs(np.linalg.inv(K-w[i]*w[i]*M+1j*w[i]*C)@(F[:]*np.exp(1j*Phase[:])))
    Dephasage[1,:]=np.angle(np.linalg.inv(K-w[i]*w[i]*M+1j*w[i]*C)@(F[:]*np.exp(1j*Phase[:])))

# plot des amplitudes en fonction de la fréquence de forçage
plt.semilogy(w, A[:,0], 'b-', label=r'$|Z_M|_{th}$')
plt.semilogy(w, A[:,1], 'r-', label=r'$|Z_N|_{th}$')
plt.semilogy(w, A[:,2], 'g-', label=r'$|Z_A|_{th}$')
plt.semilogy(w, A[:,3], 'k-', label=r'$|\theta_A|_{th}$')
plt.semilogy(omega, Amax[0], 'ob', label=r'$|Z_M|_{num}$')
plt.semilogy(omega, Amax[1], 'or', label=r'$|Z_N|_{num}$')
plt.semilogy(omega, Amax[2], 'og', label=r'$|Z_A|_{num}$')
plt.semilogy(omega, Amax[3], 'ok', label=r'$|\theta_A|_{num}$')
plt.xlabel(r'$\omega_\mathrm{forcage}$', fontsize=11)
plt.ylabel(r'Fontions de transfert dimensionnelles', fontsize=11)
plt.legend(loc='best')
plt.savefig('amp_f_transfert.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# PARTIE 2 
R = np.block([
    [np.zeros((Nddl,Nddl)), K ],
    [K, C],
])
S = np.block([
    [-K, np.zeros((Nddl, Nddl))],
    [np.zeros((Nddl,Nddl)),M]
])
print( "R et S", R, S)


# On calcule phi et lambda et on en déduit tout le reste
LAMBDAAG, PHIAG = la.eig(R,S)
print("LAMBDA Amortissement Général : Les valeurs propres")
print(LAMBDAAG)
print("PHI Amortissement Général normalisée (norme euclidienne)")
print(PHIAG)


# Extraire les parties réelles et imaginaires des valeurs propres
wn = np.abs(LAMBDAAG)  # Fréquence propre ω_n
zeta = np.real(LAMBDAAG) / wn  # Calcul du taux d’amortissement
wd = wn * np.sqrt(1 - zeta**2)  # Fréquence propre amortie
print("wn : ", wn)
print("zeta : ", zeta)
print("wd : ", wd)


# print("Matrices Abaissement d'ordre")
RD = PHIAG.transpose()@R@PHIAG
SD = PHIAG.transpose()@S@PHIAG
PHIBG=PHIAG@np.sqrt(np.linalg.inv(SD))
SDun = PHIBG.transpose()@S@PHIBG
RDlambda = PHIBG.transpose()@R@PHIBG
# print(np.diag(RD))
print('S identité')
print(SDun)
print('R Lambda')
print(RDlambda)
print('PHI Tilde')
print(PHIBG)
print("RD lambda")
print(RDlambda)



invSD = np.linalg.inv(SDun)
# number of time points
n = 50001
# time points
t = np.linspace(0,50,n)
# initial condition
zd0 = np.zeros((2*Nddl), dtype=np.complex128)
# storage
rd = np.zeros((n,2*Nddl),dtype=np.complex128)
q = np.zeros((n,2*Nddl),dtype=np.complex128)
# Forcing term
zero = [0,0,0,0]
FU = np.concatenate((zero, F[:]*np.exp(1.j*(Phase[:]))),axis=None)
FD = PHIBG.transpose()@FU
print("FD")
print(FD)
# record initial conditions
rd[0,:] = zd0[:]
# function that returns dz/dt
def modelc(tt, zz, invSS, RR, FR):
    u = np.zeros((2*Nddl), dtype=np.complex128)
    u[:] = FR[:]*np.exp(1.j*omega*tt)
    dzdt = invSS @ (-RR@zz+u)
    return dzdt
tspan = [t[0], t[n-1]]
z = solve_ivp(modelc, tspan, zd0, method='RK45', args=(invSD,RDlambda,FD),t_eval=t[0:n-1])
rd=z.y.transpose()
for i in range(n-1):
    q[i,:] = PHIBG@rd[i,:]
# Réponse du système dans le plan physique
plt.rc('text', usetex=False)
plt.plot(t[0:n], np.real(q[:,0]), 'b-', linewidth=0.5,label=r'$Z_M$')
plt.plot(t[0:n], np.real(q[:,1]), 'r-', linewidth=0.5,label=r'$Z_N$')
plt.plot(t[0:n], np.real(q[:,2]), 'g-', linewidth=0.5,label=r'$Z_A$')
plt.plot(t[0:n], np.real(q[:,3]), 'k-', linewidth=0.5,label=r'$\theta_A$')
plt.xlabel(r't (s)', fontsize=11)
plt.xlim([0, 15])
plt.ylabel(r'Déplacement (m)', fontsize=11)
plt.legend(loc='best')
plt.savefig('reponse_forcee_physique.png', dpi = 300, bbox_inches='tight')
plt.show()
# #plot
# plt.semilogy(t[0:n],np.abs(q[:,0]), 'b-', linewidth=0.5,label=r'$Z_M$')
# plt.semilogy(t[0:n],np.abs(q[:,1]), 'r-', linewidth=0.5,label=r'$Z_N$')
# plt.semilogy(t[0:n],np.abs(q[:,2]), 'g-', linewidth=0.5,label=r'$Z_A$')
# plt.semilogy(t[0:n],np.abs(q[:,3]), 'k-', linewidth=0.5,label=r'$\theta_A$')
# plt.xlabel(r't (s)', fontsize=11)
# plt.ylabel(r'$|X|_{physique}$ seulement déplacement', fontsize=11)
# plt.legend(loc='best')
# plt.show()



