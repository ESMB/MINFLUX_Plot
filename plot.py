#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:35:25 2022

@author: Mathew
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from skimage import filters,measure

image_size=10000
eps_threshold=100
minimum_locs_threshold=2

pathlist=[]


pathlist.append(r"/Users/Mathew/Documents/Edinburgh Code/MinFlux/test/data.txt")

# This is for drawing 2D Gaussians

def gkern(l,sigx,sigy):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx)/np.square(sigx) + np.square(yy)/np.square(sigy)) )
    # print(np.sum(kernel))
    # test=kernel/np.max(kernel)
    # print(test.max())
    return kernel/np.sum(kernel)

# This generates images

def generate_SR_prec(xcoord,ycoord):
    box_size=10
    SR_prec_plot_def=np.zeros((image_size,image_size),dtype=float)
    SR_points=np.zeros((image_size,image_size),dtype=float)

    for x,y in zip(xcoord,ycoord):

      
        precisionx=3
        precisiony=3
        scale_xcoord=round(x)
        scale_ycoord=round(y)

        
        
        
        tempgauss=gkern(2*box_size,precisionx,precisiony)
        
        # SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
        
        
        
        ybox_min=scale_ycoord-box_size
        ybox_max=scale_ycoord+box_size
        xbox_min=scale_xcoord-box_size
        xbox_max=scale_xcoord+box_size 
        
        SR_points[scale_xcoord,scale_ycoord]+=1
        if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
            SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss

    
    return SR_prec_plot_def,SR_points

def generate_SR_prec_cluster(x_coord,y_coord,clusters,cluster_contents,clus_low,clus_high):
    box_size=10
    SR_prec_plot_def=np.zeros((image_size,image_size),dtype=float)
    SR_fwhm_plot_def=np.zeros((image_size,image_size),dtype=float)

    j=0
    for clu,num,x,y in zip(clusters,cluster_contents,x_coord,ycoord):
        if clu>-1:
            if num>clus_low:
                if num<clus_high:
                    precisionx=3
                    precisiony=3
       
                    scale_xcoord=round(x)
                    scale_ycoord=round(y)
                    
                    sigmax=precisionx
                    sigmay=precisiony
                    
                    
                    # tempgauss=SRGaussian((2*box_size,2*box_size), (sigmax,sigmay),(box_size,box_size))
                    tempgauss=gkern(2*box_size,sigmax,sigmay)
                    ybox_min=scale_ycoord-box_size
                    ybox_max=scale_ycoord+box_size
                    xbox_min=scale_xcoord-box_size
                    xbox_max=scale_xcoord+box_size 
                
                
                    if(np.shape(SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempgauss)):
                        SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_prec_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+tempgauss
                        
                    tempfwhm_max=tempgauss.max()
                    tempfwhm=tempgauss>(0.5*tempfwhm_max)
                    
                    tempfwhm_num=tempfwhm*(clu+1)
                   
                    
                    if(np.shape(SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max])==np.shape(tempfwhm)):
                       plot_temp=np.zeros((2*box_size,2*box_size),dtype=float)
                       plot_add=np.zeros((2*box_size,2*box_size),dtype=float)
                       plot_temp=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]
                       plot_add_to=plot_temp==0
                       
                       plot_add1=plot_temp+tempfwhm_num
                       
                       plot_add=plot_add1*plot_add_to
                       
                       SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]=SR_fwhm_plot_def[ybox_min:ybox_max,xbox_min:xbox_max]+plot_add
                        
                        
                        # (SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm_num).where(SR_fwhm_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]==0)
                        # SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]=SR_tot_plot_def[scale_ycoord-box_size:scale_ycoord+box_size,scale_xcoord-box_size:scale_xcoord+box_size]+tempfwhm
                    
                    # SR_tot_plot_def[SR_tot_plot_def==0]=1
            
            labelled=SR_fwhm_plot_def
            
            SR_prec_plot=SR_prec_plot_def
            labelled=labelled      
                 
                
            
            j+=1
    
    return SR_prec_plot_def,labelled,SR_fwhm_plot_def
 

# Cluster analysis

def cluster(coords):
     db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(coords)
     labels = db.labels_
     n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
     print('Estimated number of clusters: %d' % n_clusters_)
     return labels


for path in pathlist:
    # Load the data
    data = pd.read_table(path,header=None,sep=',')
    
    
    # Multiple by 10^9 to get in nm units
    xcoord=data[0]*1e9
    ycoord=data[1]*1e9
    
    # Subtract the minimum position away to only have positive values
    minimum_x=min(xcoord)
    minimum_y=min(ycoord)
    xcoord=xcoord-minimum_x
    ycoord=ycoord-minimum_y
    
    # Generate and save SR images
  
    precplot,pointplot=generate_SR_prec(xcoord,ycoord)
    
    imsr = Image.fromarray(precplot)
    imsr.save(path+'SR_width_python.tif')
    
    imsr = Image.fromarray(pointplot)
    imsr.save(path+'SR_point_python.tif')
    
    # Perform clustering
    coords=np.array(list(zip(xcoord,ycoord)))

    clusters=cluster(coords)
    
    # Check how many localisations per cluster
     
    cluster_list=clusters.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function. 
    maximum=max(cluster_list)+1  
    
    
    cluster_contents=[]         # Make a list to store the number of clusters in
    
    for i in range(0,maximum):
        n=cluster_list.count(i)     # Count the number of times that the cluster number i is observed
       
        cluster_contents.append(n)  # Add to the list. 
    
    if len(cluster_contents)>0:
        average_locs=sum(cluster_contents)/len(cluster_contents)
 
        plt.hist(cluster_contents, bins = 20,range=[1,20], rwidth=0.9,color='#607c8e') # Plot a histogram. 
        plt.xlabel('Localisations per cluster')
        plt.ylabel('Number of clusters')
        plt.savefig('path+Localisations.pdf')
        plt.show()
        
        cluster_arr=np.array(cluster_contents)
    
        median_locs=np.median(cluster_arr)
        mean_locs=cluster_arr.mean()
        std_locs=cluster_arr.std()
    

    SR_clu_prec,labelled,SR_prec_plot=generate_SR_prec_cluster(xcoord,ycoord,clusters,cluster_contents,-1,100)
    
    imsr = Image.fromarray(SR_clu_prec)
    imsr.save(path+'clustered_width.tif')
    
    imsr = Image.fromarray(labelled)
    imsr.save(path+'clustered_labelled.tif')

    SR_dim_prec,dim_labelled,SR_dim_prec_plot=generate_SR_prec_cluster(xcoord,ycoord,clusters,cluster_contents,1,3)
    
    imsr = Image.fromarray(SR_dim_prec)
    imsr.save(path+'Dimer_width.tif')
    
    imsr = Image.fromarray(dim_labelled)
    imsr.save(path+'dim_labelled.tif')

    