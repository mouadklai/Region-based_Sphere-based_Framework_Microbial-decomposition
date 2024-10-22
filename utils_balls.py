import heapq
import random
from joblib import Parallel
from numba import jit,njit,objmode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pts 
import scipy.sparse as sparse 
import scipy.sparse.linalg as splinalg
from tqdm import tqdm
import time
import numba as nb





#Transformation
@njit(parallel=True)
def Asynchronous_Transformation(BallsSet,initialDistribution,deltat,rho, mu, rho_m,vfom,vsom,vdom,kab):
    N = BallsSet.shape[0]
    X = np.empty_like(initialDistribution)
    for i in nb.prange(N):
        # Now we are in the ball Let's initialize the variables and set some parameters
        r = BallsSet[i,3] # radius of the ball
        x1 = initialDistribution[i,0] # variable for the microbial biomass MB which is the population living inside the ball at time t
        x2 = initialDistribution[i,1] # variable for dissolved organic matter DOM at time t
        x3 = initialDistribution[i,2] # variable for soil organic matter SOM at time t
        x4 = initialDistribution[i,3] # variable for fresh organic matter FOM at time t
        x5 = initialDistribution[i,4] # variable for carbon dioxide at time t
        #  we update the parameters if and only if there is microorganisms in the ball 
        # (obvious; it's microorganism that is responsible for the transformation of the organic matter)
        if x1 > 0 : 
            #first we let the microrganisms eat some from the dom in orther to grow
            if x2>0 : 
                cDOM = 3*x2 / (4*np.pi*(r**3))  #concentration of DOM
                temp =  vdom*cDOM*x1*deltat/(kab+cDOM) 
                if x1 >= temp  : # that the microorganisms have excess DOM 
                    x1 += temp # we let the microorganisms grow
                    x2 -= temp
                else : 
                    x1 += x2 # the microorganisms don't have enough DOM in order to grow during deltat
                    x2 = 0 # it lasts no DOM anymore in this ball
            # the decomposition of MB after dying to DOM and FOM 
            temp = mu*x1*deltat #the portion of MB to be decomposed 
            if x1 >= temp : # there is enough MB
                x1 -= temp # MB dying
                x2 += rho_m * temp #fast decomposition
                x3 += (1-rho_m) * temp #slow decomposition
            else : 
                x2 += rho_m * x1 #fast decomposition
                x3 += (1-rho_m) * x1 #slow decomposition
                x1 = 0 # it lasts no MB anymore in this ball
            # the respiration of microorganisms
            if x1 > 0 : 
                temp = rho * x1 * deltat
                if x1 >= temp : 
                    x1 -= temp 
                    x5 += temp # CO2 emission by microorganisms
                else :
                    x5 +=x1 # CO2 emission by microorganisms
                    x1 = 0
        #Transformation of SOM and FOM to DOM 
        #Transformation of SOM
        if x3 >0 :
            temp = vsom * x3 * deltat # portion of SOM that can be dissolved during deltat (SOM to DOM)
            if x3 >= temp: # there is enough SOM in the ball
                x2 += temp
                x3 -= temp
            else : 
                x2 +=x3
                x3 = 0
        #transformation of FOM 
        if x4 > 0 : 
            temp = vfom * x4 * deltat # portion of FOM that can be dissolved during deltat  (FOM to DOM)
            if x4 >= temp :  # there is enough FOM in the ball
                x2 += temp 
                x4 -= temp
            else : 
                x2 += x4 
                x4 = 0
        X[i,0] = x1
        X[i,1] = x2
        X[i,2] = x3
        X[i,3] = x4
        X[i,4] = x5
    return X




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


def Simulate_explicit(BallsSet,Theta,initialDistribution,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab):
    TotalDistribution = np.sum(initialDistribution,axis=0)*100/np.sum(initialDistribution)
    history = [TotalDistribution]
    X = initialDistribution.copy()
    explicit = sparse.identity(BallsSet.shape[0]) + Dc*dt*(-Theta)@sparse.diags(1/(4/3*np.pi*BallsSet[:,3]**3))
    print("Performing asynchrone Transformation with",deltat*60*60*24*1000," ms and explicit diffusion with time step= ",
          dt*60*60*24*1000,"ms and a Diffusion coefficient Dc=", Dc,
          'for',DeltaT,"days,\n")
    with tqdm(total=int(DeltaT/deltat)) as pbar:
        start = time.time()
        for _ in range(int(DeltaT/deltat)):
            #transformation
            X = Asynchronous_Transformation(BallsSet,X,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab) 
            #Diffusion
            for _ in range(int(deltat/dt)):
                X[:,1] = explicit@X[:,1]
            # save the history
            TotalDistribution = np.sum(X,axis=0)*100/np.sum(X)
            history.append(TotalDistribution)
            pbar.update(1)
    timetaken = time.time()-start
    print("time taken = ", timetaken,"seconds")
    return history,X,timetaken

@njit
def diffuse_OM(setofBalls,massofBalls,Adjacency,Dc,dt):
    dm = np.zeros(massofBalls.shape[0],dtype=np.float64)
    for i,j in Adjacency:
        vi = 4/3*np.pi*setofBalls[i,3]**3
        vj = 4/3*np.pi*setofBalls[j,3]**3
        rij = min(setofBalls[i,3],setofBalls[j,3])
        sij = np.pi * (rij**2)
        dij = np.sqrt((setofBalls[i,0]-setofBalls[j,0])**2+(setofBalls[i,1]-setofBalls[j,1])**2+(setofBalls[i,2]-setofBalls[j,2])**2)
        cj = massofBalls[j]/vj
        ci =massofBalls[i]/vi
        flow_i_j = Dc*dt*sij*(ci - cj)/dij
        dm[i] -= flow_i_j
        dm[j] += flow_i_j
    return dm

def simulate_explicit_OM(BallsSet,Adjacency,initialDistribution,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab):
    TotalDistribution = np.sum(initialDistribution,axis=0)*100/np.sum(initialDistribution)
    history = [TotalDistribution]
    X = initialDistribution.copy()
    print("Performing asynchrone Transformation with",deltat*60*60*24*1000," ms and explicit diffusion (OM) with time step= ",
          dt*60*60*24*1000,"ms and a Diffusion coefficient Dc=", Dc,
          'for',DeltaT,"days,\n")
    with tqdm(total=int(DeltaT/deltat)) as pbar:
        start = time.time()
        for _ in range(int(DeltaT/deltat)):
            #transformation
            X = Asynchronous_Transformation(BallsSet,X,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab) 
            #Diffusion
            for _ in range(int(deltat/dt)):
                X[:,1] += diffuse_OM(BallsSet,X[:,1],Adjacency,Dc,dt)
            # save the history
            TotalDistribution = np.sum(X,axis=0)*100/np.sum(X)
            history.append(TotalDistribution)
            pbar.update(1)
    timetaken = time.time()-start
    print("time taken = ", timetaken,"seconds")
    return history,X




def Implicit_scheme(Adjacency,BallsSet,Dc,deltat):
    N= BallsSet.shape[0]
    R,C,V = [i for i in nb.prange(N)],[i for i in nb.prange(N)],[1 for i in nb.prange(N)]
    Volume = sparse.csr_matrix((np.array(V), (np.array(R), np.array(C))), shape=(N,N))
    Volume_inv = sparse.csr_matrix((1/np.array(V), (R, C)), shape=(N,N))
    #Matrix A with CSR 
    for i,j in Adjacency:
        vi = 4/3*np.pi*BallsSet[i,3]**3
        vj = 4/3*np.pi*BallsSet[j,3]**3
        dij = np.sqrt(np.sum((BallsSet[i,:3] - BallsSet[j,:3])**2))
        rij = min(BallsSet[i,3],BallsSet[j,3])
        sij = np.pi * (rij**2)
        thetaij = -Dc*deltat*sij/dij
        R.append(i)
        C.append(j)
        V.append(thetaij/vj)
        R.append(j)
        C.append(i)
        V.append(thetaij/vi)
        V[i] += -thetaij/vi
        V[j] += -thetaij/vj
    A = sparse.csr_matrix((np.array(V), (np.array(R), np.array(C))), shape=(N,N))
    M_inv = sparse.csr_matrix((1/np.array(V[:N]), (np.array(R[:N]), np.array(C[:N]))), shape=(N,N))  
    return A,M_inv


def Simulate_implicit(A,M_inv,BallsSet,initialDistribution,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab):
    TotalDistribution = np.sum(initialDistribution,axis=0)*100/np.sum(initialDistribution)
    history = [TotalDistribution]
    X = initialDistribution.copy()
    print("Performing asynchrone Transformation and Implicit diffusion with time step= ",
          dt*60*60*24*1000,"ms and transformation with time step= ",
          deltat*60*60*24*1000,"ms a Diffusion coefficient Dc=", Dc,
          'for',DeltaT,"days,\n bilan de mass",np.sum(X,axis=0))
    with tqdm(total=int(DeltaT/deltat)) as pbar:
        start = time.time()
        for _ in  nb.prange(int(DeltaT/deltat)):
            #transformation
            X = Asynchronous_Transformation(BallsSet,X,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab) 
            #Diffusion 
            #X[:,1] = splinalg.cg(A,X[:,1])[0]
            for _ in range(int(deltat/dt)):
                X[:,1] =  gradient_conjugue_preconditionne(A.data,A.indices,A.indptr,M_inv.data,M_inv.indices,M_inv.indptr,X[:,1],X[:,1],1e-7,max_itr=1000)
            # save the history
            TotalDistribution = np.sum(X,axis=0)*100/np.sum(X)
            history.append(TotalDistribution)
            pbar.update(1)
        timetaken = time.time()-start
    print("time taken = ", timetaken,"seconds")
    return history,X




def VisualizeSimulation(simulations,dt,filename,title="Simulations"):
    simulations = np.array(simulations)
    x = np.linspace(0,simulations.shape[0]*dt,simulations.shape[0])
    figure, ax = plt.subplots()
    ax.plot(x, simulations[...,0],  label='MB')
    ax.plot(x, simulations[...,1],  label='DOM')
    ax.plot(x, simulations[...,2],  label='SOM')
    ax.plot(x, simulations[...,3],  label='FOM')
    ax.plot(x, simulations[...,4], label='CO2')
    _ = ax.legend(loc='center right',shadow=True)
    figure.patch.set_facecolor('white')
    figure.suptitle(title)
    figure.savefig("Plots and Results/"+filename)
    np.savetxt('Plots and Results/'+filename,simulations)
    plt.show()




def fill_Balls_with_MB_Distributeoverthenearest(BallsSet,voxels_spots,DOM_mass):
    output = np.zeros((BallsSet.shape[0],5))
    totalVolume = np.sum(4/3*np.pi*BallsSet[:,3]**3)
    for spot in voxels_spots:
        index = (spot[0] - BallsSet[:,0])**2 + (spot[1] - BallsSet[:,1])**2 + (spot[2] - BallsSet[:,2])**2 - BallsSet[:,3]**2
        index = np.where(index == index.min())[0] 
        V = np.sum(4/3*np.pi*BallsSet[index,3]**3)
        for j in index:
            output[j,0] +=4/3*np.pi*BallsSet[j,3]**3 /V * spot[3]
    for i in  nb.prange(BallsSet.shape[0]):
        v = 4/3*np.pi*BallsSet[i,3]**3
        output[i,1] = DOM_mass*v/totalVolume
    return output




def fil_balls(BallsSet,VoxelsBiomass,DOM_mass):
    output = np.zeros((BallsSet.shape[0],5))
    totalVolume = np.sum(4/3*np.pi*BallsSet[:,3]**3)
    for i in  nb.prange(BallsSet.shape[0]):
        v = 4/3*np.pi*BallsSet[i,3]**3
        output[i,1] = DOM_mass*v/totalVolume
        indices = np.where((VoxelsBiomass[:,0] - BallsSet[i,0])**2 + (VoxelsBiomass[:,1] - BallsSet[i,1])**2 + (VoxelsBiomass[:,2] - BallsSet[i,2])**2 <= BallsSet[i,3]**2)[0]
        for j in indices:
            output[i,0]+=VoxelsBiomass[j,3]
    return output


def savearray(X,fileName):
    np.savetxt("Plots and Results/"+fileName,X)

def loadarray(fileName,dty=None):
    if dty==None:
        return np.loadtxt(fileName)
    return np.loadtxt(fileName,dtype=dty)


## intercorrelation 
def intercorrelation(f,g):
    return np.sum(f*g)/(np.sqrt(np.sum(f**2))*np.sqrt(np.sum(g**2)))


@njit
def sparseProduct(Values,Colomns,RowPtr,Vector):
    """" compressed sparse row """
    N = Vector.shape[0]
    product = np.zeros(N,dtype=np.float64)
    i=0
    while i<N:
        k=RowPtr[i]
        while k<RowPtr[i+1]:
            product[i] +=Values[k]*Vector[Colomns[k]]
            k+=1
        i+=1
    return product

@njit
def scalarProduct(vector1,vector2):
    result = 0
    N = vector1.shape[0]
    for k in range(N):
        result+=vector1[k]*vector2[k]
    return result

@njit(fastmath=True)
def iterate_CG(Values,Colomns,RowPtr,xk, rk, pk):
        """
        Basic iteration of the conjugate gradient algorithm

        Parameters:
        xk: current iterate
        rk: current residual
        pk: current direction
        """

        # construct step size
        αk = scalarProduct(rk,rk) / scalarProduct(pk,sparseProduct(Values,Colomns,RowPtr,pk))

        # take a step in current conjugate direction
        xk_new = xk + αk * pk

        # construct new residual
        rk_new = rk + αk * sparseProduct(Values,Colomns,RowPtr,pk)

        # construct new linear combination
        betak_new = scalarProduct(rk_new,rk_new) /scalarProduct(rk,rk)

        # generate new conjugate vector
        pk_new = -rk_new + betak_new * pk

        return xk_new, rk_new, pk_new

@njit(fastmath=True)
def conjugateGradient(Values,Colomns,RowPtr,b,x0,epsilon,max_iter = 2000):
    # initial iteration
    xk = x0
    rk = sparseProduct(Values,Colomns,RowPtr,xk)-b
    pk=-rk
    k=0
    while k<max_iter : 
        # run conjugate gradient iteration
        xk, rk, pk = iterate_CG(Values,Colomns,RowPtr,xk, rk, pk)
        # compute absolute error and break if converged
        err = np.linalg.norm(rk)
        if err < epsilon:
            break
        k+=1
    print("converged after ", k , " iterations")
    return xk


@njit
def gradient_conjugue_preconditionne(Values,Colomns,RowPtr,MinvValues,MinvColomns,MinvRowPtr,b,u0,tolerance,max_itr=1000):
    uk=u0
    rk = b - sparseProduct(Values,Colomns,RowPtr,uk)
    zk = sparseProduct(MinvValues,MinvColomns,MinvRowPtr,rk)
    pk = zk
    k=0
    while k<max_itr :
        rk,zk,pk,uk = iterer_GCP(Values,Colomns,RowPtr,MinvValues,MinvColomns,MinvRowPtr,rk,zk,pk,uk)
        if np.sum((rk)**2)< tolerance**2:
            return uk
        k+=1
    return b

@njit
def iterer_GCP(Values,Colomns,RowPtr,MinvValues,MinvColomns,MinvRowPtr,rk,zk,pk,uk):
    alpha_k = scalarProduct(rk,zk)/scalarProduct(sparseProduct(Values,Colomns,RowPtr, pk),pk)
    uk_new = uk+alpha_k*pk
    rk_new = rk - alpha_k*sparseProduct(Values,Colomns,RowPtr, pk)
    zk_new = sparseProduct(MinvValues,MinvColomns,MinvRowPtr, rk_new)
    beta_k = scalarProduct(rk_new,zk_new)/scalarProduct(rk,zk)
    pk_new = zk_new + beta_k * pk
    return rk_new, zk_new,pk_new,uk_new




### Distribution of biology for scenarios modeling
@nb.jit()
def homogeneousDOM_heterogenousMO(BallsSet,voxels_spots,DOM_mass):
    """
    This function takes in geometrical parameters of the balls and spots of Microorganisms from valerie's file and distribute it randomly in the balls  
    and a Total mass of dom and distribute it in a way to have the same concentration in all the pore space
    """
    output = np.zeros((BallsSet.shape[0],5))
    concentration = DOM_mass/np.sum(4/3*np.pi*BallsSet[:,3]**3)
    for spot in voxels_spots:
        n = random.randint(0,BallsSet.shape[0])
        output[n,0] += spot[3]
    for i in  nb.prange(BallsSet.shape[0]):
        v = 4/3*np.pi*BallsSet[i,3]**3
        output[i,1] = concentration*v
    return output

def heterogenousDOM_heterogenousMO(BallsSet,voxels_spots,DOM_mass,DOM_indices):
    """
    This function takes in geometrical parameters of the balls and spots of Microorganisms from valerie's file and distribute it randomly in the balls  
    and a Total mass of dom and distribute it in a way to have the same concentration in all the pore space
    """
    output = np.zeros((BallsSet.shape[0],5))
    concentration = DOM_mass/np.sum(4/3*np.pi*BallsSet[DOM_indices,3]**3)
    for spot in voxels_spots:
        n = random.randint(0,BallsSet.shape[0])
        output[n,0] += spot
    for i in DOM_indices:
        v = 4/3*np.pi*BallsSet[i,3]**3
        output[i,1] = concentration*v
    return output






##### Dijkstra and Accessibility 

@nb.jit()
def dijkstra_original(V,E,v0):
    length = V.shape[0]
    U = [i for i in nb.prange(length)]  #unvisited vertices 
    S = []  #visited vertices
    lamda = [1e+20 for i in nb.prange(length)]  #{src:0}
    lamda[v0] = 0
    while U: 
        vc = lamda.index(min([lamda[i] for i in U]))
        if vc not in U:
            return lamda
        U.remove(vc)
        S.append(vc)
        for v in U:
            distanceij = ((V[vc,0]-V[v,0])**2+(V[vc,1]-V[v,1])**2+(V[vc,2]-V[v,2])**2)**(1/2)
            if [vc,v] in E and lamda[v]>lamda[vc]+ distanceij:
                lamda[v] = lamda[vc]+ distanceij
    return lamda


def dijkstra(graph,starting_vertex):
    distances = [float('infinity') for _ in graph]
    distances[starting_vertex] = 0

    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances



def dijkstra_with_path(graph,starting_vertex):
    distances = [(float('infinity'),None) for _ in graph]
    distances[starting_vertex] = (0,starting_vertex)

    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)

        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex][0]:
            continue

        for neighbor, weight,_ in graph[current_vertex]:
            distance = current_distance + weight

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor][0]:
                distances[neighbor] = (distance,current_vertex)
                heapq.heappush(pq, (distance, neighbor))
    return distances


def get_path_cost(balls_set,starting_vertex,distances):
    mean_volume = 4/3 * np.pi* balls_set[:,3]**3
    sum_volume = 4/3 * np.pi* balls_set[:,3]**3
    for end in range(len(distances)):
        if distances[end][0] == float('infinity'):
            continue
        path = [end]
        while path[-1] != starting_vertex:
            path.append(distances[path[-1]][1])
        path.reverse()
        mean_volume[end] = 4/3 * np.pi*np.sum(balls_set[path,3]**3)
        sum_volume[end] = mean_volume[end]/len(path)
    return mean_volume,sum_volume
    


def process_graph_distance(setofBalls,Adjacency):
    graph = [[] for i in range(setofBalls.shape[0])]
    for i,j in Adjacency:
        dij = np.sqrt(np.sum((setofBalls[i,:3]-setofBalls[j,:3])**2))
        graph[i].append([j,dij])
        graph[j].append([i,dij])
    return graph

def process_graph_flux(setofBalls,Adjacency):
    graph = [[] for i in range(setofBalls.shape[0])]
    for i,j in Adjacency:
        dij = np.sqrt(np.sum((setofBalls[i,:3]-setofBalls[j,:3])**2))
        rij = min(setofBalls[i,3],setofBalls[j,3])
        sij = np.pi * (rij**2)
        graph[i].append([j,dij/sij])
        graph[j].append([i,dij/sij])
    return graph


def process_graph_surface(setofBalls,Adjacency):
    graph = [[] for i in range(setofBalls.shape[0])]
    for i,j in Adjacency:
        #dij = np.sqrt(np.sum((setofBalls[i,:3]-setofBalls[j,:3])**2))
        rij = min(setofBalls[i,3],setofBalls[j,3])
        sij = np.pi * (rij**2)
        graph[i].append([j,1/sij])
        graph[j].append([i,1/sij])
    return graph

def accessibility(BallsSet,Adjacency,initialDistribution,type="d"):
    nbnoeud = BallsSet.shape[0]
    acc = [0] * nbnoeud
    if type == "d":
        graph = process_graph_distance(BallsSet,Adjacency)
    elif type == "s":
        graph = process_graph_surface(BallsSet,Adjacency)
    elif type == 'f':
        graph = process_graph_flux(BallsSet,Adjacency) 
    for i in range(nbnoeud):
        m_MB_i = initialDistribution[i,0]
        if m_MB_i > 0: # if there is microorganisms
            # dijkstra return all the shortest paths from all the nodes of the graph to the input node 
            shortest = np.array(dijkstra(graph, i))
            for j in range(nbnoeud):
                m_DOM_j = initialDistribution[j,1]
                if m_DOM_j > 0:
                    shj = shortest[j]
                    if shj == 0:
                        if type =="d":
                            shj = BallsSet[j,3]
                        elif type == "s":
                            shj = np.pi * (BallsSet[j,3]**2)
                        elif type == "f" : 
                            shj = np.pi *BallsSet[j,3]
                    if shj < float('infinity') :
                        acc[j] +=  m_DOM_j / shj
            #acc[i] *= m_MB_i
    totalDOM = np.sum(initialDistribution[:,1])
    return np.sum(acc)/totalDOM





def calculaccess(BallsSet,Adjacency,initialDistribution):
    nbnoeud = BallsSet.shape[0]
    acc = [[0]*6] * nbnoeud
    sommemj = 0
    massebio = 0
    nbbio = 0
    nbdom = 0
    for i in range(nbnoeud):
        sommemj += initialDistribution[i,1]
        massebio += initialDistribution[i,0]
        if initialDistribution[i,1] != 0:
            nbdom += 1
        if initialDistribution[i,0] != 0:
            nbbio += 1
    print(f"\n biomasse totale avant calcul  = {massebio} ; masse totale de matiere organique avant calcul = {sommemj} ; nb boules biomasse = {nbbio} ; nb boules DOM = {nbdom}")
    nbbio = 0
    massebio = 0
    graph = process_graph_distance(BallsSet,Adjacency)
    barycentre = np.array([0,0,0],dtype=BallsSet.dtype)
    indices_bio = []
    for i in range(nbnoeud):
        if initialDistribution[i,0] > 0: # if there is microorganisms
            barycentre+=BallsSet[i,:3]
            indices_bio.append(i)
            nbbio += 1
            massebio += initialDistribution[i,0]
            # dijkstra return all the shortest paths from all the nodes of the graph to the input node 
            shortest = dijkstra(graph, i)
            for j in range(nbnoeud):
                mj = initialDistribution[j,1]
                if mj > 0:
                    shj = shortest[j]
                    if shj == 0:
                        shj = BallsSet[i,3] #rayon
                    elif shj < float('infinity') :
                        acc[j][0] +=  mj / shj #(shj+mj) #Original = mj / shj
                        acc[j][1] += 1 / (1 + np.exp(-acc[j][0]))
                        acc[j][2] += mj / shj*initialDistribution[i,0]
                        acc[j][3] += 1 / (1 + np.exp(-acc[j][2]))
                        acc[j][4] += shj /mj
                        acc[j][5] += 1 / (1 + np.exp(-acc[j][4]))
    access = [0]*6
    #new_acc = 0
    dispersion = 0
    barycentre /= len(indices_bio)
    for i in range(nbnoeud):
        if i in indices_bio:
            dispersion += np.sqrt(np.sum((BallsSet[i,:3]-barycentre)**2))
        access[0] += acc[i][0]
        access[1] += acc[i][1]
        access[2] += acc[i][2]
        access[3] += acc[i][3]
        access[4] += acc[i][4]
        access[5] += acc[i][5]
        #new_acc += 1/(1+acc[i])
    access[0] = access[0]/sommemj
    access[1] = access[1]/sommemj
    access[2] = access[2]/sommemj
    access[3] = access[3]/sommemj
    access[4] = access[4]/sommemj
    access[5] = access[5]/sommemj
    return access, dispersion






def Simulate_implicit1(A,M_inv,BallsSet,initialDistribution,DeltaT,Dc,deltat,dt,rho,mu,rho_m,vfom,vsom,vdom,kab):
    TotalDistribution = np.sum(initialDistribution,axis=0)
    history = [TotalDistribution]
    X = initialDistribution.copy()
    for _ in  nb.prange(int(DeltaT/deltat)):
        #transformation
        X = Asynchronous_Transformation(BallsSet,X,deltat,rho,mu,rho_m,vfom,vsom,vdom,kab) 
        #Diffusion 
        #X[:,1] = splinalg.cg(A,X[:,1])[0]
        for _ in range(int(deltat/dt)):
            X[:,1] =  gradient_conjugue_preconditionne(A.data,A.indices,A.indptr,M_inv.data,M_inv.indices,M_inv.indptr,X[:,1],X[:,1],1e-7,max_itr=1000)
        # save the history
        TotalDistribution = np.sum(X,axis=0)
        history.append(TotalDistribution)
    return history,X