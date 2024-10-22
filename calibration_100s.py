import calib
import numpy as np 
import matplotlib.pyplot as plt 
#import Calibration



deltat = 10/(60*60*24)
#dt = 0.3/(60*60*24)
DELTAT = 1.783/24 #0.5/24 #
Dc = 100950


print("start simulation using regions")

### Initial Distribution in the plans (Valerie) 
Z_profiles_100 = np.loadtxt('Data/table_DOC_profileZ_BD12_Sw100_15000.txt')[:,0]

#Regions and adjacency
regions = np.loadtxt('Data/new_regions/regions.txt')
voisins = np.loadtxt("Data/new_regions/voisins.txt")

regions_minmax_plans = np.loadtxt("Data/new_regions/region_z_limits.txt")
indices = np.union1d(np.loadtxt("Data/new_regions/z_newplans/z1.txt"),np.loadtxt("Data/new_regions/z_newplans/z2.txt"))


Mass_Distribution = calib.fill_regions(regions,regions_minmax_plans,indices.astype(int),DOM_mass=592759.3)

initial_state_regions = calib.regions_to_plans(regions,regions_minmax_plans,Mass_Distribution)/1000

theta = calib.THETA(voisins,regions)

finale_state_regions = calib.simulateDiffusion(regions,theta,Mass_Distribution,Dc,deltat,DELTAT)

finale_state_plans = calib.regions_to_plans(regions,regions_minmax_plans,finale_state_regions)

finale_state_plans/=1000

intercorrelation = calib.intercorrelation(finale_state_plans[:300],Z_profiles_100)

print(f"initial : {np.sum(initial_state_regions)}")

print(f"final : {np.sum(finale_state_plans)}")
print(f"valerie : {np.sum(Z_profiles_100)}")
print("Simulation using regions Done! \n intercorrelation = ",intercorrelation)

# plot results

#print("start simulation using balls")


# Load the balls
#G_100s = np.loadtxt('Data/boules-100_.bmax') # En micrometer
#Adj_100s = np.loadtxt('Data/boules-100_.adj',dtype=int)
#B_100_Calibration = Calibration.PlansToBalls(G_100s,[0,1],592759.3)





#Theta_min_100s = Calibration.THETA(Adj_100s,G_100s,surfaceRadius='min')
#B_100_finale = Calibration.simulateDiffusion(G_100s,Theta_min_100s,B_100_Calibration,0.6,Dc,deltat,DELTAT)
#B_100_finale = Calibration.BallsToPlans(G_100s,B_100_finale)

#B_100_finale/=1000

B_100_finale=np.loadtxt("Data/simulatin_balls")
intercorrelation_1 = calib.intercorrelation(B_100_finale[:300],Z_profiles_100)

print(" intercorrelation for balls = ",intercorrelation_1)






x = np.arange(0,300,1)
figure, ax = plt.subplots()
ax.plot(x,Z_profiles_100 , label='LBIOS')
ax.plot(x, B_100_finale[:300],  label='MOSAIC with balls')
ax.plot(x, finale_state_plans[:300],  label='MOSAIC with regions')
_ = ax.legend(loc='center right',shadow=True)
figure.patch.set_facecolor('white')
figure.suptitle("LBIOS vs MOSAIC_regions diffusion for 100% saturation ")
figure.savefig("LBIOS vs MOSAIC_regions 100s")
plt.show()
