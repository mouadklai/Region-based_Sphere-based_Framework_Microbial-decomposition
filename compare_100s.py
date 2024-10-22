import numpy as np
from utils import intercorrelation
import matplotlib.pyplot as plt 
deltat = 30/(60*60*24)

LBM_V = np.loadtxt("Data/Comparison/100%.txt")[:4]

Implicit_MK10s = np.loadtxt("Plots and Results/0.45/plt_5days_100s_implicit")
print(Implicit_MK10s.shape)

Implicit_MK_new = np.zeros((121,5))

j=0
for k in range(Implicit_MK10s.shape[0]):
    if k%120==0:
        Implicit_MK_new[j] = Implicit_MK10s[k]
        j+=1
Implicit_MK_new=Implicit_MK_new[:-1]
print(Implicit_MK_new.shape)
Y = np.linspace(0,Implicit_MK_new.shape[0],Implicit_MK_new.shape[0])

figure, ax = plt.subplots()
ax.plot(Y, Implicit_MK_new[...,0],'b',  label='MB MOSAIC')
ax.plot(Y, Implicit_MK_new[...,1],'g',  label='DOM MOSAIC')
ax.plot(Y, Implicit_MK_new[...,2],'c',  label='SOM MOSAIC')
ax.plot(Y, Implicit_MK_new[...,4],'r', label='CO2 MOSAIC')

ax.plot(Y, LBM_V[0],'b-.',  label='MB LBM')
ax.plot(Y, LBM_V[1],'g-.',  label='DOM LBM')
ax.plot(Y, LBM_V[2],'c-.',  label='SOM LBM')
ax.plot(Y, LBM_V[3],'r-.', label='CO2 LBM')


#ax.plot(Y, Oldimplicit[...,0],'b-.',  label='MB implicit')
#ax.plot(Y, Oldimplicit[...,1],'g-.',  label='DOM implicit')
#ax.plot(Y, Oldimplicit[...,2],'c-.',  label='SOM implicit')
#ax.plot(Y, Oldimplicit[...,4],'r-.', label='CO2 implicit')

_ = ax.legend(loc='center right',shadow=True)
figure.patch.set_facecolor('white')
figure.suptitle("LBM vs MOSAIC for 100% saturation \n Dc = 0.45x100950 ",fontdict={'size':15})
plt.xlabel("Time in hours",fontdict={'size':13} )
plt.ylabel("Total Mass in %",fontdict={'size':13})

plt.legend(bbox_to_anchor=(1, 1) , loc='center left')
figure.savefig("Plots and Results/0.45/compwithLBM")
plt.show()



"""




#print(Explicit_MK.shape)

#Explicit_MK_new = np.zeros((120,5))
Implicit_MK_new = np.zeros((120,5))

j=0
for k in range(Implicit_MK.shape[0]):
    if k%120==0:
        #print(k)
        #Explicit_MK_new[j] = Explicit_MK[k]
        Implicit_MK_new[j] = Implicit_MK[k]
        j+=1


    
Y = np.linspace(0,LBM_V[0].shape[0],LBM_V[0].shape[0])

INTERCORR_CO2 = intercorrelation( Implicit_MK_new[...,4],LBM_V[3])
INTERCORR = 1/4*(intercorrelation(Implicit_MK_new[...,0],LBM_V[0])+intercorrelation( Implicit_MK_new[...,1],LBM_V[1])+intercorrelation( Implicit_MK_new[...,2],LBM_V[2])+INTERCORR_CO2)


figure, ax = plt.subplots()
ax.plot(Y, Implicit_MK_new[...,0],'b',  label='MB MOSAIC')
ax.plot(Y, Implicit_MK_new[...,1],'g',  label='DOM MOSAIC')
ax.plot(Y, Implicit_MK_new[...,2],'c',  label='SOM MOSAIC')
ax.plot(Y, Implicit_MK_new[...,4],'r', label='CO2 MOSAIC')

ax.plot(Y, LBM_V[0],'b-.',  label='MB LBM')
ax.plot(Y, LBM_V[1],'g-.',  label='DOM LBM')
ax.plot(Y, LBM_V[2],'c-.',  label='SOM LBM')
ax.plot(Y, LBM_V[3],'r-.', label='CO2 LBM')
_ = ax.legend(loc='center right',shadow=True)
figure.patch.set_facecolor('white')
figure.suptitle("Implicit vs LBM for 100% saturation \n Dc = 60_000 \n mean of intercorrelation between all curves ="+ str(INTERCORR)+"%\n mean of intercorrelation for carbon curve ="+ str(INTERCORR_CO2)+"%",fontdict={'size':15})
plt.xlabel("Time in hours",fontdict={'size':13} )
plt.ylabel("Total Mass in %",fontdict={'size':13})

plt.legend(bbox_to_anchor=(1, 1) , loc='center left')
figure.savefig("Plots and Results/newImplicit60000/newImplicit Dc 100_950_vs_LMB_100s")
plt.show()


#figure, ax = plt.subplots()
#ax.plot(Y, Implicit_MK_new[...,0],'b',  label='MB')
#ax.plot(Y, Implicit_MK_new[...,1],'g',  label='DOM')
#ax.plot(Y, Implicit_MK_new[...,2],'c',  label='SOM')
#ax.plot(Y, Implicit_MK_new[...,4],'r', label='CO2')

#ax.plot(Y, LBM_V[0],'b-.',  label='MB')
#ax.plot(Y, LBM_V[1],'g-.',  label='DOM')
#ax.plot(Y, LBM_V[2],'c-.',  label='SOM')
#ax.plot(Y, LBM_V[3],'r-.', label='CO2')
#_ = ax.legend(loc='center right',shadow=True)
#figure.patch.set_facecolor('white')
#figure.suptitle("Implicit vs LBM for 100% saturation")

#plt.legend(bbox_to_anchor=(0.95, 0.4) , loc='center left')
#figure.savefig("Plots and Results/Implicit_vs_LMB_100s")

"""