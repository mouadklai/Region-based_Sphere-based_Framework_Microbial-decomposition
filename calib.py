import numpy as np
import scipy.sparse as sparse 
import scipy.sparse.linalg as spLinalg



def THETA(voisins,regions):
    theta = sparse.lil_matrix((regions.shape[0],regions.shape[0]))
    for i,j,sij in voisins:
        dij = np.sqrt(np.sum((regions[int(i),:3] - regions[int(j),:3])**2))
        theta[int(i),int(j)] = -sij/dij
        theta[int(j),int(i)] = -sij/dij
        theta[int(i),int(i)] -= theta[int(i),int(j)]
        theta[int(j),int(j)] -= theta[int(j),int(i)]
    return theta



def simulateDiffusion(regions,Theta,DOM_distribution,Dc,deltat,DELTAT):
    implicit = sparse.identity(regions.shape[0]) + Dc*deltat*Theta@sparse.diags(1/regions[:,3])
    X = DOM_distribution
    for i in range(int(DELTAT/deltat)):
        X = spLinalg.cg(implicit,X)[0]
        if i%1000==0:
            print(i)
    return X




def fill_regions(regions,limits_regions,indices,DOM_mass):
    output = np.zeros((regions.shape[0]))
    #indexes = np.empty(0,dtype=int)
    #for i in plans:
    #    lower = np.where(limits_regions[:,0]<i)[0]
    #    higher = np.where(limits_regions[:,1]>i)[0]
    #    indexes = np.union1d(indexes,np.intersect1d(lower,higher))
    #print(indexes)
    selectedregionsvolume = np.sum(regions[indices,3])
    output[indices]= DOM_mass*regions[indices,3]/selectedregionsvolume
    return output



def regions_to_plans(regions,minmaxregions,DOM_distribution):
    output = np.zeros((512))
    for i in range(regions.shape[0]):
        plans = np.arange(minmaxregions[i,0],minmaxregions[i,1]+1,1) #+int((minmaxregions[i,1]-minmaxregions[i,0])/2)
        for z0 in plans:
            output[int(z0)] += DOM_distribution[i]/plans.shape[0]
    return output


def intercorrelation(f,g):
    return np.sum(f*g)/(np.sqrt(np.sum(f**2))*np.sqrt(np.sum(g**2)))
