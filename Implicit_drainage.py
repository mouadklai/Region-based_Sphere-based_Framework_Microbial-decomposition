import utils
import scipy.sparse as sparse 
import numpy  as np


if __name__ == "__main__":
    #Parameters 
    voxelHeight = 24 #micrometer
    microorganismWeight = 5.4*10**(-8) #microgramme
    tolerance = 10-5 # tolerance for the conjugate gradient
    Dc = 0.5*100950 #40_000 # micrometer².j⁻¹
    deltat = 30/(60*60*24) # step time in days for implicit scheme and transformation process 
    dt = 15/(60*60*24) # step time in days for Explicit scheme
    DeltaT= 5 #Time step in days of the hole simulation 
    [rho,mu,rho_m,vfom,vsom,vdom,kab] = [float(a) for a in open("Data/boules.par").readlines()[0].split(" ")] 
    kab*=voxelHeight**3

    print('rho =',rho,' , mu =',mu,' , rho_m =',rho_m,' , vfom =',vfom,' , vsom =',vsom,' , vdom =',vdom,' , kab =',kab)
    print('filling balls with microorganisms and organic matter')

    DOM_totalMass = 289.5 #microgramme
    VoxelsBiomass = utils.loadarray("Data/biomasse.dat")
    VoxelsBiomass[:,3]*= microorganismWeight



    regions = np.loadtxt("Data/first_regions/regions.txt")
    voisins = np.loadtxt("Data/first_regions/voisins.txt")

    initial_condition = utils.fill_Balls_with_MB_Distributeoverthenearest(regions,VoxelsBiomass,DOM_totalMass)

    A,M_inv = utils.Implicit_scheme(voisins,regions,Dc,dt)

    history_100s_implicit, X_5days_100s_implicit = utils.Simulate_implicit(A,M_inv,regions,initial_condition,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab)


    utils.savearray(X_5days_100s_implicit,"0.45/X_5days_100s_implicit")
    
    utils.VisualizeSimulation(history_100s_implicit,deltat,'0.45/plt_5days_100s_implicit','100% saturation \nDc = '+str(Dc))