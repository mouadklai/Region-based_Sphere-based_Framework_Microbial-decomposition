import numpy as np
import scipy.sparse as sparse 
import scipy.sparse.linalg as spLinalg
import random

def cubes_and_adjacency(L, dx):
    num_cubes_per_side = int(L/dx)
    cubes = []
    cubes_adjacency = []
    for i in range(num_cubes_per_side):
        for j in range(num_cubes_per_side):
            for k in range(num_cubes_per_side):
                x = (i+0.5)*dx
                y = (j+0.5)*dx
                z = (k+0.5)*dx
                cubes.append([x, y, z,dx**3])
                # Check if cube has adjacent cubes and add their indices to cube_adjacency
                if i > 0:
                    cubes_adjacency.append([i*num_cubes_per_side*num_cubes_per_side + j*num_cubes_per_side + k,
                                            (i-1)*num_cubes_per_side*num_cubes_per_side + j*num_cubes_per_side + k,
                                            dx*dx,
                                            dx])
                if j > 0:
                    cubes_adjacency.append([i*num_cubes_per_side*num_cubes_per_side + j*num_cubes_per_side + k,
                                            i*num_cubes_per_side*num_cubes_per_side + (j-1)*num_cubes_per_side + k,
                                            dx*dx,
                                            dx])
                if k > 0:
                    cubes_adjacency.append([i*num_cubes_per_side*num_cubes_per_side + j*num_cubes_per_side + k,
                                            i*num_cubes_per_side*num_cubes_per_side + j*num_cubes_per_side + k-1,
                                            dx*dx,
                                            dx])
    return np.array(cubes), np.array(cubes_adjacency)




def THETA(cubes,adjacency):
    theta = sparse.lil_matrix((cubes.shape[0],cubes.shape[0]))
    for i,j,sij,dij in adjacency:
        theta[int(i),int(j)] = -sij/dij
        theta[int(j),int(i)] = -sij/dij
        theta[int(i),int(i)] -= theta[int(i),int(j)]
        theta[int(j),int(j)] -= theta[int(j),int(i)]
    return theta


def full_THETA(cubes,adjacency):
    theta = sparse.lil_matrix((cubes.shape[0],cubes.shape[0]))
    for i,j,sij,dij in adjacency:
        theta[int(i),int(j)] = -sij/dij
        theta[int(i),int(i)] -= theta[int(i),int(j)]
    return theta


def simulateDiffusion(cubes,Theta,initial_distribution,Dc,deltat,DELTAT):
    implicit = sparse.identity(cubes.shape[0]) + Dc*deltat*Theta@sparse.diags(1/cubes[:,3])
    X = initial_distribution
    for i in range(int(DELTAT/deltat)):
        X = spLinalg.cg(implicit,X)[0]
    return X


def fill_irregular_cubes(cubes,z_limit,DOM_mass,z_limits):
    output = np.zeros((cubes.shape[0]))
    total_volume_of_cubes = 0
    for i in range(cubes.shape[0]):
        if z_limits[i,1] <= z_limit : 
            total_volume_of_cubes += cubes[i,3]
            output[i] = DOM_mass*cubes[i,3]
    output /= total_volume_of_cubes
    return output

def fill_cubs(cubes,z_limit,DOM_mass):
    output = np.zeros((cubes.shape[0]))
    indices = np.where(cubes[:,2]<=z_limit)[0]
    total_volume_of_cubes = np.sum(cubes[indices,3])
    output[indices]= DOM_mass*cubes[indices,3]/total_volume_of_cubes
    return output


def irregular_cubs_to_z_plans(cubes,DOM_distribution,z_limits,nb_plans):
    output = np.zeros((nb_plans))
    for i in range(cubes.shape[0]):
        plans = np.arange(z_limits[i,0],z_limits[i,1]+1,1) 
        for z0 in plans:
            output[int(z0)] += DOM_distribution[i]/plans.shape[0]
    return output


def cubs_to_z_plans(cubes,dx,DOM_distribution):
    min_cubes_limits = cubes[:,2]-dx*0.5
    max_cubes_limits = cubes[:,2]+dx*0.5
    max_z = np.max(max_cubes_limits)
    nb_plans =  round(max_z,0)
    output = np.zeros((int(nb_plans)))
    for i in range(cubes.shape[0]):
        plans = np.arange(min_cubes_limits[i],max_cubes_limits[i],1)
        for z0 in plans:
            output[int(z0)] += DOM_distribution[i]/plans.shape[0]
    return output

def groupe_cubes(cubes,adjacency,indices):
    #indices must be of adjacent nodes 
    nodes = np.hstack((np.array([[i] for i in range(cubes.shape[0])]),cubes))
    edges = []
    vertices = []
    x,y,z=0,0,0
    v=0
    for j in indices:
        x += cubes[j,0]
        y += cubes[j,1]
        z += cubes[j,2]
        v+=cubes[j,3]
        nodes[j,1:] = -1
    x/= len(indices)
    y/= len(indices)
    z/= len(indices)
    nodes = np.vstack((nodes,np.array([nodes.shape[0],x,y,z,v])))
    for i,j,sij,dij in adjacency:
        if int(i) in indices and int(j) not in indices  :
            newi = int(nodes[-1,0])
            newSij = min(nodes[-1,4]**(1/3),nodes[int(j),4]**(1/3))**2
            print(min(nodes[-1,4]**(1/3),nodes[int(j),4]**(1/3)))
            newDij = np.sqrt(np.sum((cubes[int(j),:3]-nodes[-1,1:4])**2))
            edges.append([newi,int(j),newSij,newDij])
        elif int(j) in indices and int(i) not in indices :             
            newj = int(nodes[-1,0])
            newSij = min(nodes[-1,4]**(1/3),nodes[int(i),4]**(1/3))**2
            print(min(nodes[-1,4]**(1/3),nodes[int(i),4]**(1/3)))
            newDij = np.sqrt(np.sum((cubes[int(i),:3]-nodes[-1,1:4])**2))
            edges.append([int(i),newj,newSij,newDij])
        elif int(j) not in indices and int(i) not in indices : 
            edges.append([int(i),int(j),sij,dij])
    for k,x,y,z,v in nodes:
        if x!=-1:
            vertices.append([k,x,y,z,v])
    edges = np.array(edges)
    vertices = np.array(vertices)
    indexes = vertices[:,0].tolist()
    for k in range(len(edges)):
        edges[k] = [indexes.index(edges[k,0]),indexes.index(edges[k,1]),edges[k,2],edges[k,3]]
    return vertices[:,1:],np.array(edges)


def dividecube(L,l):
    ## Create a 3D image of length L
    threeD_image = np.zeros(3*(L,),dtype=int)
    index = 1
    for i in range(int(threeD_image.shape[0]/l)):
        for j in range(int(threeD_image.shape[0]/l)):
            for k in range(int(threeD_image.shape[0]/l)):
                threeD_image[i*l:(i+1)*l,j*l:(j+1)*l,k*l:(k+1)*l] = index
                index+=1
    return threeD_image


def randomly_slice_3D_image(threeD_image,length_big_slice):
    is_list = [i for i in range(int(threeD_image.shape[0]/length_big_slice)-1)]
    js_list = [i for i in range(int(threeD_image.shape[0]/length_big_slice)-1)]
    ks_list = [i for i in range(int(threeD_image.shape[0]/length_big_slice)-1)]
    for dx in [1,2,3]:
        index = np.max(threeD_image)
        i = random.choice(is_list)
        j = random.choice(js_list)
        k = random.choice(ks_list)
        is_list.remove(i)
        js_list.remove(j)
        ks_list.remove(k)
        threeD_image[i*length_big_slice:(i+1)*length_big_slice,
                     j*length_big_slice:(j+1)*length_big_slice,
                     k*length_big_slice:(k+1)*length_big_slice] = dividecube(length_big_slice,dx)+index
    return threeD_image

def graph_from_3D_image(threeD_image):
    image_length_per_side = threeD_image.shape[0]
    z_limits = []
    nodes = []
    regions = np.unique(threeD_image).tolist()
    A = {}
    for region in regions:
        A[region] = {}
        is_list, js_list, ks_list = np.where(threeD_image==region)
        centerx,centery,centerz = 0,0,0
        for i,j,k in zip(is_list, js_list, ks_list):
            centerx += i+0.5
            centery += j+0.5
            centerz += k+0.5
            if i == min(is_list) and i>0 :
                if threeD_image[i-1,j,k] in A[region]:
                    A[region][threeD_image[i-1,j,k]] +=1
                else :
                    A[region][threeD_image[i-1,j,k]] =1
            if i == max(is_list) and i <image_length_per_side-1 :
                if threeD_image[i+1,j,k] in A[region]:
                    A[region][threeD_image[i+1,j,k]] +=1
                else :
                    A[region][threeD_image[i+1,j,k]] =1
            if j == min(js_list) and j>0 :
                if threeD_image[i,j-1,k] in A[region]:
                    A[region][threeD_image[i,j-1,k]] +=1
                else :
                    A[region][threeD_image[i,j-1,k]] =1
            if j == max(js_list) and j<image_length_per_side-1 :
                if threeD_image[i,j+1,k] in A[region]:
                    A[region][threeD_image[i,j+1,k]] +=1
                else :
                    A[region][threeD_image[i,j+1,k]] =1
                    
            if k == min(ks_list) and k>0 :
                if threeD_image[i,j,k-1] in A[region]:
                    A[region][threeD_image[i,j,k-1]] +=1
                else :
                    A[region][threeD_image[i,j,k-1]] =1
            if k == max(ks_list) and k<image_length_per_side-1 :
                if threeD_image[i,j,k+1] in A[region]:
                    A[region][threeD_image[i,j,k+1]] +=1
                else :
                    A[region][threeD_image[i,j,k+1]] =1
        centerx /= is_list.shape[0]
        centery /= js_list.shape[0]
        centerz /= ks_list.shape[0]
        z_limits.append([min(ks_list),max(ks_list)])
        nodes.append([centerx,centery,centerz,is_list.shape[0]])
    nodes = np.array(nodes)
    adjacency = []
    for region in A.keys() : 
        for adj_region in A[region].keys() : 
            i,j = regions.index(region),regions.index(adj_region)
            dij = np.sqrt(np.sum((nodes[i,:3]-nodes[j,:3])**2))
            adjacency.append([i,j,A[region][adj_region],dij])
    return nodes, np.array(adjacency) , np.array(z_limits)




def intercorrelation(f,g):
    return np.sum(f*g)/(np.sqrt(np.sum(f**2))*np.sqrt(np.sum(g**2)))