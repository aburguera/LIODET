#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_liodet
# Description : liodet usage example
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 15-Nov-2024 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from pickle import load,dump
import openslide
from liodet import globalParameters,get_blobs,plot_blobs

###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################

# =============================================================================
# PLOT_MASKS
#
# Plots the masks in a matrix of nCols columns. The area of these masks, provi-
# ded in areaList, is printed inside the mask. This is NOT meant to be a gene-
# ral purpose plot function. It is only to help with this example.
#
# Input  : maskList - List of binary masks (the masks to plot)
#          areaList - List of integers (the corresponding areas)
# =============================================================================
def plot_masks(maskList,areaList,nCols):
    # Compute the number of rows needed.
    nRows=(len(maskList)+nCols-1)//nCols
    # Create the figure with subplots arranged in a grid of nRows x nCols
    theFigure,theAxes=plt.subplots(nRows,nCols,figsize=(nCols*2,nRows*2))
    # Flatten axes for easier indexing, in case nRows or nCols is 1
    theAxes=theAxes.ravel() if isinstance(theAxes,np.ndarray) else [theAxes]
    # Plot the images
    for i in range(len(maskList)):
        # Display each image and remove axes
        theAxes[i].imshow(maskList[i],cmap='prism')
        theAxes[i].text(maskList[i].shape[1]/2,maskList[i].shape[0]/2, '%d'%areaList[i], horizontalalignment='center',verticalalignment='center')
        theAxes[i].axis('off')
    # Turn off remaining empty subplots
    for j in range(len(maskList),len(theAxes)):
        theAxes[j].axis('off')
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

###############################################################################
# MAIN CODE
###############################################################################

# Modify parameters according to your needs. Let's use NDPI level 3 to have a
# fast execution (though the default, level 0, is what should be used).
globalParameters['NDPI_LEVEL_HIRES']=3

# Load the NDPI file
ndpiData=openslide.OpenSlide('../../DATA/CMU-1.ndpi')

# Run the main function to extract blobs and segmentation masks.
theBlobs=get_blobs(ndpiData,globalParameters)

# Save the blobs to a file to avoid re-computing them later again.
with open('SAVED_BLOBS.pkl','wb') as outFile:
    dump(theBlobs,outFile)

# Load them (just to exemplify how to do it)
with open('SAVED_BLOBS.pkl','rb') as inFile:
    theBlobs=load(inFile)

# Plot the results in level 5 (to have a small picture)
theImage=plot_blobs(ndpiData,theBlobs,3,globalParameters)
plt.figure(figsize=(20,20))
plt.imshow(theImage)
plt.show()

# Filter the correct blobs (with info code !=2). Keep only the first
# item (curBlob[0]), which is the blob.
correctBlobs=[curBlob[0] for curBlob in theBlobs if curBlob[-1][-1]!=2]

# Compute some features. Just as an example, compute the area
theAreas=[np.sum(curBlob) for curBlob in correctBlobs]

# Now show the first 10 blobs
numBlobsToShow=min(10,len(correctBlobs))
blobsToShow=[curBlob for curBlob in correctBlobs[:numBlobsToShow]]
areasToShow=theAreas[:numBlobsToShow]

# Plot them
plot_masks(blobsToShow,areasToShow,5)

# Filter the wrong blobs (with info code ==2) to show them.
wrongBlobs=[curBlob[0] for curBlob in theBlobs if curBlob[-1][-1]==2]

# Compute some features. Just as an example, compute the area
theAreas=[np.sum(curBlob) for curBlob in wrongBlobs]

# Now show the first 10 wrong blobs
numBlobsToShow=min(10,len(wrongBlobs))
blobsToShow=[curBlob for curBlob in wrongBlobs[:numBlobsToShow]]
areasToShow=theAreas[:numBlobsToShow]

# Plot them
plot_masks(blobsToShow,areasToShow,5)

# Close the openslide file
ndpiData.close()