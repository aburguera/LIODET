#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : liodet
# Description : The provided functions help in performing object detection in
#               very large images (i.e. images too large to be fully loaded
#               into memory).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Nov-2024 - First functional version.
#               14-Nov-2024 - Major refactoring.
#               15-Nov-2024 - Minor adjustment and aesthetic changes.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
from skimage.draw import line
from skimage.util import img_as_float
from skimage.filters import gaussian,threshold_otsu
from skimage.segmentation import flood_fill
from skimage.morphology import remove_small_objects,remove_small_holes,disk
from skimage.measure import label, regionprops
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt,binary_closing
from scipy.sparse.csgraph import connected_components

###############################################################################
# GLOBAL PARAMETERS
#
# This dictionary provides configuration settings for processing NDPI images,
# including parameters for image resolution levels, crop dimensions, segmenta-
# tion thresholds, and bounding box processing.
#
# The parameters whose name ends with LEVEL0 are scale dependant and the provi-
# ded value is for NDPI level 0. Before using them, they have to be scaled
# to the desired resolution level. The function "convert_parameters" does it.
###############################################################################

globalParameters = {
    'NDPI_LEVEL_LORES': 7,                  # NDPI level for coarse data check
    'NDPI_LEVEL_HIRES': 0,                  # NDPI level for fine image proc.
    'CROP_SIZE_LEVEL0': (4000, 4000),       # Base crop size
    'GAUSSIAN_BLUR_SIGMA_LEVEL0': 5,        # Sigma for Gaussian smoothing
    'SMALL_OBJECT_SIZE_LEVEL0': 1000,       # Min area for small objects
    'SMALL_HOLE_SIZE_LEVEL0': 1000,         # Min area for small holes
    'CLOSING_RADIUS_LEVEL0': 20,            # Radius for morphological closing
    'DELTA_OVERLAP_LEVEL0': 512,            # Max distance for bbox contiguity
    'MAX_TOUCH_RATIO': 0.15,                # Max allowed ratio of edge pixels
    'MIN_AREA_LEVEL0': 4000,                # Min area for objects of interest
    'FLOOD_FILL_START': (0, 0),             # Seed point for background fill
    'GRID_CROP_MARGIN': 0.1                 # Margin to enlarge each crop area
}

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# =============================================================================
# PROGRESS_BAR
#
# Displays a progress bar in the console. When curValue>=maxValue, a newline is
# printed so that further texts (progress bars or not) go to the next line.
#
# Input  : theText  : str, label for progress bar.
#          curValue : int, current progress value.
#          maxValue : int, total value for completion.
# =============================================================================
def progress_bar(theText,curValue,maxValue):
    thePercentage=curValue/maxValue
    curSize=int(50*thePercentage)
    sys.stdout.write('\r')
    sys.stdout.write("%s : [%-50s] %d%%"%(theText,'='*curSize,int(thePercentage*100)))
    sys.stdout.flush()
    if curValue>=maxValue:
        print()

###############################################################################
# COORDINATE AND PARAMETER CONVERSION FUNCTIONS
###############################################################################

# =============================================================================
# CONVERT_COORDINATES
#
# Converts coordinates between specified NDPI image levels.
#
# Input  : srcCoordinates : ndarray, (r,c) coordinates at the source level.
#          srcLevel       : int, source NDPI level.
#          dstLevel       : int, destination NDPI level.
#          ndpiData       : OpenSlide object with NDPI data.
#
# Output : ndarray, (r,c) coordinates scaled to dstLevel.
# =============================================================================
def convert_coordinates(srcCoordinates,srcLevel,dstLevel,ndpiData):
    return np.round(np.array(srcCoordinates)*(ndpiData.level_downsamples[srcLevel]/ndpiData.level_downsamples[dstLevel])).astype('int')

# =============================================================================
# CONVERT_PARAMETERS
#
# Scales parameters to the destination NDPI level from Level 0. Length-related
# parameters (such as crop region -width and height-, structuring element
# radius, ...) the parameter is linearly scaled depending on the source and
# destination level scaling factors. Area-realted parameters (object and hole
# sizes, minimum areas, ...) are scaled quadratically. Check the code.
#
# Input  : dstLevel        : int, destination NDPI level.
#          globalParameters: dict, configuration parameters.
#          ndpiData        : OpenSlide object with NDPI data.
#
# Output : Converted parameters tuple, scaled to dstLevel. Check the code to
#          see which parameters are returned and the globalParameters
#          dictionary to learn about their meaning.
# =============================================================================
def convert_parameters(dstLevel,globalParameters,ndpiData):
    lengthMultiplier=(ndpiData.level_downsamples[0]/ndpiData.level_downsamples[dstLevel])
    areaMultiplier=lengthMultiplier*lengthMultiplier
    return (
                convert_coordinates(globalParameters['CROP_SIZE_LEVEL0'],0,dstLevel,ndpiData),
                int(globalParameters['GAUSSIAN_BLUR_SIGMA_LEVEL0']*lengthMultiplier),
                int(globalParameters['SMALL_OBJECT_SIZE_LEVEL0']*areaMultiplier),
                int(globalParameters['SMALL_HOLE_SIZE_LEVEL0']*areaMultiplier),
                int(globalParameters['CLOSING_RADIUS_LEVEL0']*lengthMultiplier),
                int(globalParameters['DELTA_OVERLAP_LEVEL0']*lengthMultiplier),
                int(globalParameters['MIN_AREA_LEVEL0']*areaMultiplier)
           )

###############################################################################
# IMAGE VISUALIZATION
###############################################################################

# =============================================================================
# PLOT_BLOBS
#
# Overlays blobs and bounding boxes on a NDPI image for visualization.
# The drawing criteria is:
#  * Blob color BLUE   - Blob is OK and there was no need to join boxes.
#  * Blob color ORANGE - Blob is OK and it is the result of joining boxes.
#  * Blob color RED    - Blob is NOT OK and should NOT be used. This blob fai-
#                        led one or more checks.
#  * Box side GREEN    - No contiguous boxes undetected at this side.
#  * Box side RED      - Contiguous boxes undetected at this side.
#
# Input  : ndpiData        : OpenSlide object with NDPI image data.
#          theCrops        : list, contains binary masks and bounding box coor-
#                            dinates.
#          ndpiLevelToPlot : int, NDPI level for display.
#          globalParameters: dict, configuration parameters.
#
# Output : theImage        : RGB image array with blob overlays.
# =============================================================================
def plot_blobs(ndpiData,theCrops,ndpiLevelToPlot,globalParameters):
    # Get the parameter
    ndpiLevelHires=globalParameters['NDPI_LEVEL_HIRES']
    # Get the image
    theImage=np.array(ndpiData.read_region(location=(0,0),level=ndpiLevelToPlot,size=ndpiData.level_dimensions[ndpiLevelToPlot]).convert('RGB'))
    # For each crop
    for curMask,(rStart,cStart,rEnd,cEnd,numTouchTop,numTouchBottom,numTouchLeft,numTouchRight,curInfo) in theCrops:
        # Convert coordinates to hi-res (since they are stored for that level)
        rStart,cStart=convert_coordinates((rStart,cStart),ndpiLevelHires,ndpiLevelToPlot,ndpiData)
        rEnd,cEnd=convert_coordinates((rEnd,cEnd),ndpiLevelHires,ndpiLevelToPlot,ndpiData)
        # Generate bounding box lines
        rRight,cRight=line(rStart,cEnd,rEnd,cEnd)
        rLeft,cLeft=line(rStart,cStart,rEnd,cStart)
        rBottom,cBottom=line(rEnd,cStart,rEnd,cEnd)
        rTop,cTop=line(rStart,cStart,rStart,cEnd)
        # Resize the mask
        targetHeight=rEnd-rStart
        targetWidth=cEnd-cStart
        if targetHeight>1 and targetWidth>1: # To prevent too small masks
            if curInfo==0:
                maskColor=[0,0,255]
            else:
                maskColor=[255,0,0] if curInfo==2 else [255,100,0]
            theImage[rStart:rEnd, cStart:cEnd][resize(curMask,(targetHeight,targetWidth),anti_aliasing=False)>0]=maskColor
        # Check theInfo boundaries fulfilled or not fulfilled
        colRight=[255,0,0] if numTouchRight>0 else [0,255,0]
        colLeft=[255,0,0] if numTouchLeft>0 else [0,255,0]
        colBottom=[255,0,0] if numTouchBottom>0 else [0,255,0]
        colTop=[255,0,0] if numTouchTop>0 else [0,255,0]
        # Draw the lines
        theImage[rRight,cRight]=colRight
        theImage[rLeft,cLeft]=colLeft
        theImage[rBottom,cBottom]=colBottom
        theImage[rTop,cTop]=colTop
    return theImage

###############################################################################
# IMAGE PROCESSING
###############################################################################

# =============================================================================
# PROCESS_IMAGE
#
# Generates a binay or ternary mask given an image. Change this function AND
# the calls to this function to use other mask generation algorithms. The
# current algorithms does:
# * Binarizes the image (green channel, which is magenta's complementary).
# * Removes small object ans holes.
# * Performs a morphological closing operation to join very close disconnected
#   regions.
# * Fills the proposed regions (flood fill) with the proposed value.  This is
#   useful to "transport" regions segmented in other scales without having to
#   scale the mask.
#
# Input  : theImage         : RGB image ndarray.
#          gaussianBlurSigma: float, sigma for Gaussian blur.
#          theThreshold     : float, binarization threshold or None to deter-
#                             mine it using Otsu method.
#          smallObjectSize  : int, min object area.
#          smallHoleSize    : int, min hole area.
#          closingRadius    : int, closing radius.
#          pixToFill        : list of (row, column) flood fill points.
#          valToFill        : int, fill value.
#
# Output : theMask          : 2D binary or ternary mask ndarray.
#          theThreshold     : Threshold used for binarization.
# =============================================================================
def process_image(theImage,gaussianBlurSigma,theThreshold,smallObjectSize,smallHoleSize,closingRadius,pixToFill,valToFill):
    # Gaussian smooth of the green channel (green is complementary to magenta)
    grayImage=gaussian(theImage[:,:,1],gaussianBlurSigma)
    # Determine the threshold if necessary
    if theThreshold is None:
        theThreshold=threshold_otsu(grayImage)
    # Binarize the image
    theMask=grayImage>theThreshold
    # Remove small objects and holes
    if smallObjectSize>0:
        theMask=remove_small_objects(theMask,smallObjectSize)
    if smallHoleSize>0:
        theMask=remove_small_holes(theMask,smallHoleSize)
    # Perform dilation followed by erosion (closing) to join small objects.
    if closingRadius>0:
        theMask=binary_closing(np.pad(theMask,pad_width=closingRadius,mode='constant',constant_values=0),structure=disk(closingRadius))[closingRadius:-closingRadius,closingRadius:-closingRadius]
    # Flood fill from the pixToFill points with a valToFill
    theMask=theMask.astype('uint8')
    for curPixToFill in pixToFill:
        theMask=flood_fill(theMask,tuple(curPixToFill),valToFill)
    return theMask,theThreshold

###############################################################################
# BOUNDING BOX AND BLOB VALIDATION
###############################################################################

# =============================================================================
# GET_CENTRAL_PIXEL_PER_BLOB
#
# Identifies the central pixel for each blob in a labeled mask. The pixel coor-
# dinates are those where a distance transform is maximized, thus guaranteeing
# that it is a pixel inside the corresponding blob.
#
# Input  : labeledMask   - 2D ndarray, labeled binary mask with each blob
#                          assigned a unique label.
#
# Output : centralPixels - ndarray of shape (N,2), where N is the number of
#                          blobs. Each row row contains the (row, column) coor-
#                          dinates of the inner pixel farthest from the
#                          boundary for each blob.
# =============================================================================
def get_central_pixel_per_blob(labeledMask):
    # Get number of blobs and initialize storage
    numBlobs=int(np.max(labeledMask))
    centralPixels=np.zeros((numBlobs,2),dtype='int')
    for idBlob in range(1,numBlobs+1):
        # Apply the distance transform on the padded current blob mask. Padding ensures outside image is considered 0
        theDistances=distance_transform_edt(np.pad(labeledMask==idBlob,pad_width=1,mode='constant',constant_values=0))
        # Find the pixel with the maximum distance value within the blob and store it
        maxDistance=np.max(theDistances)
        centralPixels[idBlob-1,:]=np.argwhere(theDistances==maxDistance)[0]
    # Adjust coordinates to padding and return
    return centralPixels-1

# =============================================================================
# COUNT_TOUCHES
#
# Counts isolated runs of 1 in a binary array. By providing to this function
# the bondary rows or columns of a binary image, it is possible to count in how
# many places the binary mask/blob toiches the image contour.
#
# Input  : theData  - 1D binary array.
#
# Output : theCount - int, number of runs of consecutive 1.
# =============================================================================
def count_touches(theData):
    theData=theData.flatten().astype('int')
    theCount=np.sum(np.diff(theData)==-1)
    if theData[-1]==1:
        theCount+=1
    return theCount

# =============================================================================
# CHECK_MASK
#
# Validates a binary mask. The checks are:
# * There is only one object (and backgroud) in the mask.
# * The ratio of image boundaries with mask pixels is small.
# * The object is large enough.
#
# Input  : theMask       - 2D binary mask array.
#          maxTouchRatio - float, max mask pixel to image boundary ratio
#          minArea       - int, min object area.
#
# Output : doAccept      - True if all checks are passed, False otherwise.
# =============================================================================
def check_mask(theMask,maxTouchRatio,minArea):
    # Check if there is exactly one foreground blob plus the background.
    # This ensures that only one blob object is detected in the mask.
    #if len(np.unique(label(theMask,connectivity=2)))!=2:
    #    return False
    # Check the ratio of the image boundary containing blob pixels (foreground).
    # If too many boundary pixels contain blob, it may indicate improper cropping at the image edges.
    if (np.sum(theMask[0,:])+np.sum(theMask[-1,:])+np.sum(theMask[:,0])+np.sum(theMask[:,-1]))/((theMask.shape[0]+theMask.shape[1])*2)>=maxTouchRatio:
        return False
    # Check if the blob area meets the minimum required size.
    # Ensures the detected blob is sufficiently large for analysis.
    if np.sum(theMask)<=minArea:
        return False
    # All checks passed; return True indicating a valid mask.
    return True

# =============================================================================
# DO_OVERLAP
#
# Determines if two bounding boxes are adjacent within a tolerance.
#
# Input  : theBox1, theBox2 - bounding boxes. A bounding box format is:
#                             * numTouchTop: Number of touches with other
#                               bounding boxes on the top side.
#                             * numTouchBottom: Number of touches with other
#                               bounding boxes on the bottom side.
#                             * numTouchLeft: Number of touches with other
#                               bounding boxes on the left side.
#                             * numTouchRight: Number of touches with other
#                               bounding boxes on the top side.
#                             * r,c : Coordinates of an inner object point
#                             * rbbStart, cbbStart, rbbEnd, cbbEnd : bounding
#                               box coordinates (row and column start and end)
#          theDelta         - int, max distance for adjacency.
#
# Output : True if they are adjacent, False otherwise.
#
# Note   : The corresponding touch count is updated.
# =============================================================================
def do_overlap(theBox1,theBox2,theDelta):
    # Define direction checks and needed parameters. Check the for loop to understand what these values are.
    theDirections=(
        (3,2,theBox1[9],theBox2[7],theBox1[6],theBox1[8],theBox2[6],theBox2[8]),  # Right of theBox1 to Left of theBox2
        (2,3,theBox1[7],theBox2[9],theBox1[6],theBox1[8],theBox2[6],theBox2[8]),  # Left of theBox1 to Right of theBox2
        (1,0,theBox1[8],theBox2[6],theBox1[7],theBox1[9],theBox2[7],theBox2[9]),  # Bottom of theBox1 to Top of theBox2
        (0,1,theBox1[6],theBox2[8],theBox1[7],theBox1[9],theBox2[7],theBox2[9])   # Top of theBox1 to Bottom of theBox2
    )
    # Check for all directions if the two boxes are contiguous
    for (firstVal,secondVal,theEnd1,theStart2,theStart1P,theEnd1P,theStart2P,theEnd2P) in theDirections:
        if theBox1[firstVal]>0 and theBox2[secondVal]>0:           # Check if continuity is compatible between boxes
            if abs(theEnd1-theStart2) <= theDelta:                 # Check if edges are within theDelta pixels
                if (theEnd1P>=theStart2P and theStart1P<=theEnd2P):# Check for overlap in the perpendicular direction
                    # Modify boundary markers to prevent further continuity detection in this direction
                    theBox1[firstVal]-=1    # Decrease the corresponding counter in theBox1
                    theBox2[secondVal]-=1   # Decrease the corresponding counter in theBox2
                    return True             # If they are contiguous, there is no need to continue checking. Return True
    # If no contiguity has been found, return False
    return False

###############################################################################
# CROP GENERATION FUNCTIONS
###############################################################################

# =============================================================================
# GRID_CROP
#
# Performs a regular grid scan of the provided image, outputing the cells of
# that grid that contain object data inside. In order to decide if a cell
# contains data or not, the image is binarized using process_image with the
# provided parameters.
#
# Input  : theImage           - RGB input image ndarray.
#          cropSize           - tuple, crop dimensions.
#          gaussianBlurSigma  - float, Gaussian blur sigma.
#          floodFillStart     - tuple, seed point for flood fill.
#          theMargin          - float, margin ratio for crop area.
#
# Output : cropList           - list of crop (rStart,cStart,rEnd,cEnd)
#          theMask            - binary mask (0 = background, 1 = data).
#          outsidePixels      - list, points inside background areas. Useful
#                               to project background into the hi-res crops.
#          theThreshold       : float, green channel binarization threshold.
# =============================================================================
def grid_crop(theImage,cropSize,gaussianBlurSigma,floodFillStart,theMargin):
    # Pick the green channel (magenta's complementary), slightly blur it, binarize
    # it with the Otsu threshold and flood fill with "2" from floodFillStart.
    theMask,theThreshold=process_image(img_as_float(theImage),gaussianBlurSigma,None,0,0,0,[floodFillStart],2)
    # Binarize again so that 0 means "outside data" and 1 means "inside data".
    theMask=(theMask<=1).astype('uint8')
    # Traverse the mask in constant intervals and keep the crops containing data.
    cropList=[]
    outsidePixels=[]
    for curRow in range(theMask.shape[0]//cropSize[0]):
        for curCol in range(theMask.shape[1]//cropSize[1]):
            # Candidate crop coordinates
            rStart=curRow*cropSize[0]
            rEnd=rStart+cropSize[0]
            cStart=curCol*cropSize[1]
            cEnd=cStart+cropSize[1]
            # Enlarged crop used for search. If data is present inside this enlarged rectangle,
            # te candidate rectangle is stored.
            rStartMargin=int(max(0,rStart-cropSize[0]*theMargin))
            cStartMargin=int(max(0,cStart-cropSize[1]*theMargin))
            rEndMargin=int(min(theMask.shape[0],rEnd+cropSize[0]*theMargin))
            cEndMargin=int(min(theMask.shape[1],cEnd+cropSize[1]*theMargin))
            # Check if there is data. Other checks ("at least xxx data pixels, ...") could be performed here.
            if theMask[rStartMargin:rEndMargin,cStartMargin:cEndMargin].any():
                # Store the crop
                cropList.append([rStart,cStart,rEnd,cEnd])
                # Store points inside the "outside-data" regions
                outsidePixels.append(get_central_pixel_per_blob(label(theMask[rStart:rEnd,cStart:cEnd]==0,connectivity=2)))
    return cropList,theMask,outsidePixels,theThreshold

###############################################################################
# MAIN IMAGE PROCESSING FUNCTION
###############################################################################

# =============================================================================
# GET_BLOBS
#
# Main function to extract blobs and generate segmentation masks.
#
# Input  : ndpiData         - OpenSlide object with NDPI image data.
#          globalParameters - dict, global parameters
#          theThreshold     - External binarization threshold. If none, it is
#                             computed here using Otsu's method.
#
# Output : outBinaryCrops   - list, processed binary mask and bounding boxes.
#                             Each list item is:
#                             [curMask,[rStart,cStart,rEnd,cEnd,numTouchTop,
#                              numTouchBottom,numTouchLeft,numTouchRight,
#                              theInfo]]
#                             where:
#                             * curMask - The binary mask.
#                             * rStart,cStart,rEnd,cEnd: Bounding box in
#                               the hi-res (according to globalPaarameters)
#                               NDPI level.
#                             * numTouchXXX - Number of unresolved adjacencies
#                               in each direction.
#                             * theInfo - Integer representing the quality of
#                               the crop. Possible values are:
#                               0 - Correct, detected during the first step.
#                               1 - Correct, detected during the second step.
#                               2 - Incorrect. Some checks in check_mask failed.
# =============================================================================
def get_blobs(ndpiData,globalParameters,theThreshold=None):
    # Get the parameters
    ndpiLevelLores=globalParameters['NDPI_LEVEL_LORES']
    ndpiLevelHires=globalParameters['NDPI_LEVEL_HIRES']
    floodFillStart=globalParameters['FLOOD_FILL_START']
    gridCropMargin=globalParameters['GRID_CROP_MARGIN']
    maxTouchRatio=globalParameters['MAX_TOUCH_RATIO']

    ###
    # FIRST STEP: BUILD AND EXPLORE A REGULAR GRID ####
    ###

    # Get the low resolution image
    loresImage=np.array(ndpiData.read_region(location=(0,0),level=ndpiLevelLores,size=ndpiData.level_dimensions[ndpiLevelLores]).convert('RGB'))
    # Convert parameters from level 0 to ndpiLevelLores
    [loresCropSize,loresGaussianBlurSigma,_,_,_,_,_]=convert_parameters(ndpiLevelLores,globalParameters,ndpiData)
    # Get the crops
    loresCropList,loresMask,loresOutPixels,cropThreshold=grid_crop(loresImage,loresCropSize,loresGaussianBlurSigma,floodFillStart,gridCropMargin)
    if theThreshold is None:
        theThreshold=cropThreshold
    # Convert crops to level 0 resolution. Only rStart,cStart are required (height and width are hiresCropSize).
    # These coordinates will be used by read_region, which requires them to be in level 0 scale.
    lvl0CropList=np.array([convert_coordinates(curCrop[:2],ndpiLevelLores,0,ndpiData) for curCrop in loresCropList])
    # Convert crops (only rStart,cStart) to hires level resolution. This information will be mainly used to have a common
    # reference frame to store partially visible blobs.
    hiresCropList=np.array([convert_coordinates(curCrop[:2],ndpiLevelLores,ndpiLevelHires,ndpiData) for curCrop in loresCropList])
    # Convert outside-data pixel coordinates to high resolution level.
    # These coordinates will be used once the hires image is loaded, so they must be scaled to that level.
    hiresOutPixels=[np.array([convert_coordinates(curPixel,ndpiLevelLores,ndpiLevelHires,ndpiData) for curPixel in curOutPixels]) for curOutPixels in loresOutPixels]
    # Convert parameters from level 0 to NDndpiLevelHires.
    [hiresCropSize,gaussianBlurSigma,smallObjectSize,smallHoleSize,closingRadius,deltaOverlap,minArea]=convert_parameters(ndpiLevelHires,globalParameters,ndpiData)
    # Init storage
    globalBoundingBoxes=[]
    outBinaryCrops=[]
    # Process all crops
    for idCrop in range(len(hiresCropList)):
        progress_bar('PROCESSING GRID ',idCrop,len(hiresCropList)-1)
        # Get the hires crop coordinates
        rStart,cStart=lvl0CropList[idCrop]
        # Get the outside-data points
        outPixels=hiresOutPixels[idCrop]
        # Get the RGB image
        rgbImage=np.array(ndpiData.read_region(location=(cStart,rStart),level=ndpiLevelHires,size=(hiresCropSize[1],hiresCropSize[0])).convert('RGB'))
        # Process the image (smooth, remove holes, floodfill from specified points and binarize)
        theMask,_=process_image(rgbImage,gaussianBlurSigma,theThreshold,smallObjectSize,smallHoleSize,closingRadius,outPixels,False)
        # Segment the mask
        theMask=label(theMask,connectivity=2)
        theRegions=regionprops(theMask)
        boundingBoxes=[curRegion.bbox for curRegion in theRegions]
        # For each detected blob:
        for idxBB,curBB in enumerate(boundingBoxes):
            # Get the mask of this blob
            curMask=(theMask==(idxBB+1))[curBB[0]:curBB[2],curBB[1]:curBB[3]].astype('uint8')
            # Check if the blob touches the image boundaries and count how many times per side.
            numTouchTop=count_touches(curMask[0,:]) if curBB[0]<=0 else 0
            numTouchBottom=count_touches(curMask[-1,:]) if curBB[2]>=(theMask.shape[0]-1) else 0
            numTouchLeft=count_touches(curMask[:,0]) if curBB[1]<=0 else 0
            numTouchRight=count_touches(curMask[:,-1]) if curBB[3]>=(theMask.shape[1]-1) else 0
            doTouch=(numTouchTop+numTouchBottom+numTouchLeft+numTouchRight)>0
            # If the blob touches the boundary, compute the coordinates of an inner pixel
            theCenter=np.array([-1,-1])
            if doTouch:
                theCenter=get_central_pixel_per_blob(curMask)[0]
                # Convert these coordinates to global
                theCenter[0]+=(curBB[0]+hiresCropList[idCrop,0])
                theCenter[1]+=(curBB[1]+hiresCropList[idCrop,1])
            # Store the bounary touches,the bounding boxes and inner point in hi-res resolution
            rbbStart,cbbStart,rbbEnd,cbbEnd=curBB[0]+hiresCropList[idCrop,0],curBB[1]+hiresCropList[idCrop,1],curBB[2]+hiresCropList[idCrop,0],curBB[3]+hiresCropList[idCrop,1]
            globalBoundingBoxes.append([numTouchTop,numTouchBottom,numTouchLeft,numTouchRight,theCenter[0],theCenter[1],rbbStart,cbbStart,rbbEnd,cbbEnd])
            # If the blob does not touch the image contour, store it for further analysis together with the bounding box coordinates
            # in hi-res scale (or analyze it here).
            if not doTouch:
                # If the mask does not have sufficient area, mark it with a 2.
                # Note that the other checks in check_mask are not performed
                # here because this mask is guaranteed to have a single blob
                # and to not have adjacent masks.
                outBinaryCrops.append([curMask,[rbbStart,cbbStart,rbbEnd,cbbEnd,numTouchTop,numTouchBottom,numTouchLeft,numTouchRight,(np.sum(curMask)<minArea)*2]])
    # Convert globalBoundingBoxes to ndarray to help further processing
    globalBoundingBoxes=np.array(globalBoundingBoxes)
    # Now analyze the global bounding boxes that touch crop boundaries
    boundaryBoundingBoxes=globalBoundingBoxes[np.sum(globalBoundingBoxes[:,:4],axis=1)>0,:]

    ###
    # SECOND STEP: MERGE CONTIGUOUS BOUNDING BOXES
    ###

    # Build the adjacency matrix
    numBoxes=boundaryBoundingBoxes.shape[0]
    adjacencyMatrix=np.zeros((numBoxes,numBoxes))
    for iFirst in range(boundaryBoundingBoxes.shape[0]-1):
        progress_bar('ADJACENCY MATRIX',iFirst,boundaryBoundingBoxes.shape[0]-2)
        for iSecond in range(iFirst+1,boundaryBoundingBoxes.shape[0]):
            if do_overlap(boundaryBoundingBoxes[iFirst],boundaryBoundingBoxes[iSecond],deltaOverlap):
                adjacencyMatrix[iFirst,iSecond]+=1
                # In theory, for non-directed graphs there is no need for this, but just in case let's do it
                adjacencyMatrix[iSecond,iFirst]+=1
    # Get the connected components
    nComp,theLabels=connected_components(adjacencyMatrix)
    # Build the groups from the connected components. Each group in the list contains the boundaryBoundingBoxes index of the boxes to join
    theGroups=[np.argwhere(theLabels==i).flatten() for i in range(nComp)]
    # Process each group
    for iGroup,curGroup in enumerate(theGroups):
        progress_bar('MERGING GROUPS  ',iGroup,len(theGroups)-1)
        # Get the current group data
        curBoundaryBB=boundaryBoundingBoxes[curGroup,:]
        # Update the boundary touches. Unused, computed only to have additional information.
        # [numTouchTop, numTouchBottom, numTouchLeft, numTouchRight, rCenter (UNUSED HERE), cCenter (UNUSED HERE), rStart, cStart, rEnd, cEnd].
        numTouchTopJ=np.sum(curBoundaryBB[:,0])
        numTouchBottomJ=np.sum(curBoundaryBB[:,1])
        numTouchLeftJ=np.sum(curBoundaryBB[:,2])
        numTouchRightJ=np.sum(curBoundaryBB[:,3])
        # Compute the group coodinates (in hi-res scale)
        rStartJ=np.min(curBoundaryBB[:,6])
        cStartJ=np.min(curBoundaryBB[:,7])
        rEndJ=np.max(curBoundaryBB[:,8])
        cEndJ=np.max(curBoundaryBB[:,9])
        # Add some margin
        theHeight=rEndJ-rStartJ
        theWidth=cEndJ-cStartJ
        rStartJ-=int(theHeight*0.1)
        rEndJ+=int(theHeight*0.1)
        cStartJ-=int(theWidth*0.1)
        cEndJ+=int(theWidth*0.1)
        rStartJ=max(rStartJ,0)
        cStartJ=max(cStartJ,0)
        rEndJ=min(rEndJ,ndpiData.level_dimensions[ndpiLevelHires][1])
        cEndJ=min(cEndJ,ndpiData.level_dimensions[ndpiLevelHires][0])
        # Compute the flood points (in hi-res scale) relative to the joint BB (since they will be when the image is cropped)
        floodPointsJ=boundaryBoundingBoxes[curGroup,4:6]-[rStartJ,cStartJ]
        # Get the image. Compute the level 0 coordinates first for the read_region location parameter
        rStartJL0,cStartJL0=convert_coordinates((rStartJ,cStartJ),ndpiLevelHires,0,ndpiData)
        theImage=np.array(ndpiData.read_region(location=(cStartJL0,rStartJL0),level=ndpiLevelHires,size=(cEndJ-cStartJ,rEndJ-rStartJ)).convert('RGB'))
        # Compute the mask. Flood fill the floodPointsJ to ensure the new decection coincides with the first one.
        curMask,_=process_image(theImage,gaussianBlurSigma,theThreshold,smallObjectSize,smallHoleSize,closingRadius,floodPointsJ,255)
        # Now keep only the "255" value
        curMask=(curMask==255).astype('uint8')
        # Check if the mask is correct.
        if check_mask(curMask,maxTouchRatio,minArea):
            # If correct, store it with the boundaryTouchJ code with bit 7=1 and bit6=0 indicating it is refined with no errors.
            outBinaryCrops.append([curMask,[rStartJ,cStartJ,rEndJ,cEndJ,numTouchTopJ,numTouchBottomJ,numTouchLeftJ,numTouchRightJ,1]])
        else:
            # If incorrect, store it with the boundaryTouchJ code with bit 7=1 and bit6=1 indicating it is refined with errors.
            outBinaryCrops.append([curMask,[rStartJ,cStartJ,rEndJ,cEndJ,numTouchTopJ,numTouchBottomJ,numTouchLeftJ,numTouchRightJ,2]])
    # Return the crops
    return outBinaryCrops