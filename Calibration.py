import numpy as np
import scipy.sparse as sparse 
import scipy.sparse.linalg as spLinalg

def PlansToBalls(BallsSet,plans,DOM_mass):
    output = np.zeros((BallsSet.shape[0]))
    indexes=np.empty(0,dtype=int)
    for i in plans:
        lower = np.where(BallsSet[:,2]-BallsSet[:,3]<i)[0]
        higher = np.where(BallsSet[:,2]+BallsSet[:,3]>i)[0]
        indexes = np.union1d(indexes,np.intersect1d(lower,higher))
    selectedBallsVolume = np.sum((4/3)*np.pi*BallsSet[indexes,3]**3)
    output[indexes]= (4/3)*DOM_mass*np.pi*(BallsSet[indexes,3]**3)/selectedBallsVolume
    return output


def BallsToPlans(BallsSet,DOM_distribution):
    output = np.zeros((512))
    for i in range(BallsSet.shape[0]):
        z= BallsSet[i,2] 
        ball_radius = BallsSet[i,3]
        plans = np.arange(int(z-ball_radius),int(z+ball_radius)+1,1)
        surface_radius = np.sqrt(ball_radius**2 + (z-plans)**2)
        j=0
        for z0 in plans:
            output[z0] += DOM_distribution[i]*surface_radius[j]/np.sum(surface_radius)
            j+=1
    return output

def simulateDiffusion(BallsSet,Theta,DOM_distribution,alpha,Dc,deltat,DELTAT):
    implicit = sparse.identity(BallsSet.shape[0]) + alpha*Dc*deltat*Theta@sparse.diags(1/(4/3*np.pi*BallsSet[:,3]**3))
    X = DOM_distribution
    for i in range(int(DELTAT/deltat)):
        X = spLinalg.cg(implicit,X)[0]
        if i%1000==0:
            print(i)
    return X
## intercorrelation 
def intercorrelation(f,g):
    return np.sum(f*g)/(np.sqrt(np.sum(f**2))*np.sqrt(np.sum(g**2)))


def THETA(Adjacency,BallsSet,surfaceRadius='min'):
    if surfaceRadius=='min':
        theta = sparse.lil_matrix((BallsSet.shape[0],BallsSet.shape[0]))
        for i,j in Adjacency:
            dij = np.sqrt(np.sum((BallsSet[i,:3] - BallsSet[j,:3])**2))
            rij = min(BallsSet[i,3],BallsSet[j,3])
            sij = np.pi * (rij**2)
            theta[i,j] = -sij/dij
            theta[j,i] = -sij/dij
            theta[i,i] -= theta[i,j]
            theta[j,j] -= theta[j,i]
        return theta
    elif surfaceRadius=='harmonicMean':
        theta = sparse.lil_matrix((BallsSet.shape[0],BallsSet.shape[0]))
        for i,j in Adjacency:
            dij = np.sqrt(np.sum((BallsSet[i,:3] - BallsSet[j,:3])**2))
            rij = 2*BallsSet[i,3]*BallsSet[j,3]/(BallsSet[i,3]+BallsSet[j,3])
            sij = np.pi * (rij**2)
            theta[i,j] = -sij/dij
            theta[j,i] = -sij/dij
            theta[i,i] -= theta[i,j]
            theta[j,j] -= theta[j,i]
        return theta
    elif surfaceRadius=='Mean':
        theta = sparse.lil_matrix((BallsSet.shape[0],BallsSet.shape[0]))
        for i,j in Adjacency:
            dij = np.sqrt(np.sum((BallsSet[i,:3] - BallsSet[j,:3])**2))
            rij = (BallsSet[i,3]+BallsSet[j,3])/2 #min(BallsSet[i,3],BallsSet[j,3])
            sij = np.pi * (rij**2)
            theta[i,j] = -sij/dij
            theta[j,i] = -sij/dij
            theta[i,i] -= theta[i,j]
            theta[j,j] -= theta[j,i]
        return theta