import utils
import scipy.sparse as sparse 
import numpy  as np


if __name__ == "__main__":
    #Parameters 
    voxelHeight = 24 #micrometer
    microorganismWeight = 5.4*10**(-8) #microgramme
    tolerance = 10-5 # tolerance for the conjugate gradient
    Dc = 60_000 #40_000 # micrometer².j⁻¹
    deltat = 30/(60*60*24) # step time in days for implicit scheme and transformation process 
    dt = 10/(60*60*24) # step time in days for Explicit scheme
    DeltaT= 5 #Time step in days of the hole simulation 
    [rho,mu,rho_m,vfom,vsom,vdom,kab] = [float(a) for a in open("DATA/boules.par").readlines()[0].split(" ")] 
    kab*=voxelHeight**3

    print('rho =',rho,' , mu =',mu,' , rho_m =',rho_m,' , vfom =',vfom,' , vsom =',vsom,' , vdom =',vdom,' , kab =',kab)
    print('filling balls with microorganisms and organic matter')

    DOM_totalMass = 289.5 #microgramme
    VoxelsBiomass = utils.loadarray("DATA/biomasse.dat")
    VoxelsBiomass[:,3]*= microorganismWeight
    print(np.max(VoxelsBiomass[:,3]),' ',np.min(VoxelsBiomass[:,3]))

    # Load the balls 
    G_100s = utils.loadarray('DATA/BallsSet/boules-100_.bmax') # in voxel
    print(DOM_totalMass/np.sum(4/3*np.pi*G_100s[:,3]**3))
    # load the adjacency of the balls
    Adj_100s = utils.loadarray('DATA/Adjacency/boules-100_.adj',int)
    # Initial distribution
    B_100s = utils.fill_Balls_with_MB_Distributeoverthenearest(G_100s,VoxelsBiomass,DOM_totalMass)
    print('Constructing the matrix')
    A,M_inv = utils.Implicit_scheme(Adj_100s,G_100s,Dc,dt)

    history_100s_implicit, X_5days_100s_implicit = utils.Simulate_implicit(A,M_inv,G_100s,B_100s,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab)
    #imp,M = utils.new_implicit(Adj_100s,G_100s,Dc,deltat)

    #history_100s_implicit, X_5days_100s_implicit = utils.Simulate_implicit2(imp,M,G_100s,B_100s,DeltaT,Dc,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab)



    utils.savearray(X_5days_100s_implicit,"X_5days_100s_implicit")
    
    utils.VisualizeSimulation(history_100s_implicit,deltat,'plt_5days_100s_implicit','100% saturation \nDc = '+str(Dc))