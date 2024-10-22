import utils
import scipy.sparse as sparse 
import numpy  as np
import os 

if __name__ == "__main__":
    #Parameters 
    print("simulation using implicit scheme     :: ")
    method = input( "Method of extraction of pore network (look in the folder donnees )    :: ")
    try: 
        os.path.isdir("donnees/"+method)
    except : 
        print("no data found! check folder 'donnees'")
        method = input( "Method of extraction of pore network (look in the folder donnees )    :: ")
    sable = input("folder in the folder 'donnees' containing the pore network (look in the folder donnees )    :: ")
    saturation = input('saturation in (%)     :: ')
    voxelHeight = 24 #micrometer
    microorganismWeight = 5.4*10**(-8) #microgramme
    tolerance = 10-5 # tolerance for the conjugate gradient
    alpha =  float(input("alpha     :: "))
    Dc = float(input('diffusion coefficient     :: '))*alpha #40_000 # micrometer².j⁻¹
    deltat = float(input('time step in    (seconds)    :: '))/(60*60*24) # step time in days for transformation process 
    dt = 15/(60*60*24)   # step time in days for implicit scheme
    DeltaT= float(input('simulation time in  ( days )   :: ')) #Time step in days of the hole simulation 
    [rho,mu,rho_m,vfom,vsom,vdom,kab] = [float(a) for a in open("donnees/boules.par").readlines()[0].split(" ")] 
    kab*=voxelHeight**3

    print('rho =',rho,' , mu =',mu,' , rho_m =',rho_m,' , vfom =',vfom,' , vsom =',vsom,' , vdom =',vdom,' , kab =',kab)
    print(f'filling {method} regions with microorganisms and organic matter')

    DOM_totalMass = 289.5 #microgramme
    VoxelsBiomass = utils.loadarray("donnees/biomasse.dat")
    VoxelsBiomass[:,3]*= microorganismWeight



    pores = np.loadtxt("donnees/"+method+"/"+sable+"/"+saturation+"%/pores.geometry")
    if method == 'balls' : 
        #load the adjacency of the balls
        adjacency = utils.loadarray("donnees/"+method+"/"+sable+"/"+saturation+"%/pores.adj",int)
        # Initial distribution
        initial_condition = utils.fill_Balls_with_MB_Distributeoverthenearest_balls(pores,VoxelsBiomass,DOM_totalMass)
        print('Constructing the matrix')
        A,M_inv = utils.Implicit_scheme_balls(adjacency,pores,Dc,dt)
    elif method == 'curvilinear skeleton' : 
        ajacency = np.loadtxt("donnees/"+method+"/"+sable+"/"+saturation+"%/pores.adj")

        initial_condition = utils.fill_Balls_with_MB_Distributeoverthenearest_regions(pores,VoxelsBiomass,DOM_totalMass)

        A,M_inv = utils.Implicit_scheme_regions(ajacency,pores,Dc,dt)

    history_100s_implicit, X_5days_100s_implicit = utils.Simulate_implicit(A,M_inv,pores,initial_condition,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab)
    path_name = "Plots and Results/"
    np.savetxt(path_name + 'His__'+method+'M'+sable+"soil"+saturation+'s_'+str(DeltaT)+'d_'+str(alpha)+'alpha',history_100s_implicit)
    np.savetxt(path_name + 'lastDis_'+method+'M'+sable+"soil"+saturation+'s_'+str(DeltaT)+'d_'+str(alpha)+'alpha',X_5days_100s_implicit)
    print("done !")
    #utils.VisualizeSimulation(history_100s_implicit,deltat,'0.45/plt_5days_100s_implicit','100% saturation \nDc = '+str(Dc))