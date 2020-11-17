import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from minimumCostPathFunc import minimumCostMask 

def apply_filter(im, filt, zero_padding=False):
    
    wdth = filt.shape[0]//2
    
    filt_img = np.zeros(im.shape)
    
    if zero_padding:
        im = np.vstack( (np.zeros( (wdth, im.shape[1]) ), im, np.zeros( (wdth, im.shape[1]) ) ) )
        im = np.hstack( (np.zeros( (im.shape[0], wdth) ), im, np.zeros( (im.shape[0], wdth) ) ) )
    else:
        im = np.vstack(([im[0]]*wdth, im, [im[-1]]*wdth))
        im = np.hstack((np.hstack([im[:,0].reshape(-1,1)]*wdth), im, np.hstack([im[:,-1].reshape(-1,1)]*wdth)))

    for i in range(wdth, im.shape[0]-wdth):
        for j in range(wdth, im.shape[1]-wdth):
            filt_img[i-wdth][j-wdth] = np.sum(im[i-wdth:i+wdth+1, j-wdth:j+wdth+1] * filt)
    
    return np.rint(filt_img)

prewitt =  [np.array([
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
        ]) , np.array([
                [ 1,  1,  1],
                [ 0,  0,  0],
                [-1, -1, -1]
        ])]

def decouple(img, type=0):
    
    img = img.astype('float64')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity_layer = np.nan_to_num(np.true_divide(((img[:, :, 0]**2) + (img[:, :, 1]**2) + (img[:, :, 2]**2)) , (img[:,:,0]+img[:, :, 1]+img[:, :, 2])))
        intensity_layer = np.rint(intensity_layer).astype('int')
        
        if type < 2:
            dx = apply_filter(intensity_layer.astype('float64'), prewitt[0])
            dy = apply_filter(intensity_layer.astype('float64'), prewitt[1])
        
            gradient = np.nan_to_num(np.arctan(dy/dx))
        else:
            return intensity_layer
    
    if type == 1:
        return gradient
    return intensity_layer,gradient


def Construct(textureImgArray, targetImgArray, blockSize, overlapSize, alpha=0.1, tolerance=0.1, finalImage=None, compareImage=None, stochastic_mask=None):
    
    texture_lum, texture_grd = decouple(textureImgArray)
    target_lum, target_grd = decouple(targetImgArray)

    try:
        compare_lum, compare_grd = decouple(compareImage)
    except:
        compare_lum, compare_grd = None, None

    # print(textureImgArray.shape, targetImgArray.shape)
    outSizeX = targetImgArray.shape[0]
    outSizeY = targetImgArray.shape[1]
    [m,n,c] = textureImgArray.shape
    
    #final image is initialised with elemnts as -1.
    try:
        finalImage_lum, finalImage_grd = decouple(finalImage)
    except:
        finalImage = np.ones([outSizeX, outSizeY, c])*-1
        finalImage_lum = np.ones([outSizeX, outSizeY], dtype='int')*-1
        finalImage_grd = np.ones([outSizeX, outSizeY], dtype='float64')*-1

    finalImage[0:blockSize[0],0:blockSize[1],:] = textureImgArray[0:blockSize[0],0:blockSize[1],:]
    finalImage_lum[0:blockSize[0],0:blockSize[1]] = texture_lum[0:blockSize[0],0:blockSize[1]]
    finalImage_grd[0:blockSize[0],0:blockSize[1]] = texture_grd[0:blockSize[0],0:blockSize[1]]

    noOfBlocksInRow = 1+np.ceil((outSizeX - blockSize[1])*1.0/(blockSize[1] - overlapSize))
    noOfBlocksInCol = 1+np.ceil((outSizeY - blockSize[0])*1.0/(blockSize[0] - overlapSize))
    
    for i in range(int(noOfBlocksInRow)):
        for j in range(int(noOfBlocksInCol)):
            if i == 0 and j == 0:
                continue
            #start and end location of block to be filled is initialised
            startX = int(i*(blockSize[0] - overlapSize))
            startY = int(j*(blockSize[1] - overlapSize))
            endX = int(min(startX+blockSize[0],outSizeX))
            endY = int(min(startY+blockSize[1],outSizeY))
            # print(startX, endX, startY, endY)

            toFill = finalImage[startX:endX,startY:endY,:]
            toFill_lum = finalImage_lum[startX:endX,startY:endY]
            toFill_grd = finalImage_grd[startX:endX,startY:endY]

            # targetBlock = targetImgArray[startX:endX,startY:endY,:]
            trg_lum = target_lum[startX:endX,startY:endY]
            trg_grd = target_grd[startX:endX,startY:endY]

            matchBlock, matchBlock_lum, matchBlock_grd, cx, cy = MatchBlock(textureImgArray, texture_lum, texture_grd, 
                                                                    toFill[:,:,0], toFill_lum, toFill_grd, 
                                                                    trg_lum, trg_grd, 
                                                                    blockSize, alpha, tolerance, stochastic_mask)
            try:
                comparePatch, comparePatch_lum, comparePatch_grd = compareImage[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]],\
                                                                   compare_lum[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]],\
                                                                   compare_grd[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]]

                targetPatch, targetPatch_lum, targetPatch_grd = targetImgArray[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]],\
                                                                target_lum[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]],\
                                                                target_lum[cx:cx+toFill.shape[0], cy:cy+toFill.shape[1]]

                diffs = np.array([SQDIFF(comparePatch_lum, comparePatch_grd, targetPatch_lum, targetPatch_grd), SQDIFF(matchBlock_lum, matchBlock_grd, targetPatch_lum, targetPatch_grd)])
                if np.argmax(diffs) == 0:
                    matchBlock, matchBlock_lum, matchBlock_grd = comparePatch, comparePatch_lum, comparePatch_grd
            except:
                pass

            B1EndY = startY+overlapSize-1
            B1StartY = B1EndY-(matchBlock.shape[1])+1
            B1EndX = startX+overlapSize-1
            B1StartX = B1EndX-(matchBlock.shape[0])+1
            if i == 0:      
                overlapType = 'v'
                B1 = finalImage[startX:endX,B1StartY:B1EndY+1,:]
                # print(B1.shape,matchBlock.shape,'v',B1StartY,B1EndY,startX,startY)
                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],0,overlapType,overlapSize)
                # print('Mask', mask.shape)
            elif j == 0:          
                overlapType = 'h'
                B2 = finalImage[B1StartX:B1EndX+1, startY:endY, :]
                # print(B2.shape,matchBlock.shape,B1StartX,B1EndY)
                mask = minimumCostMask(matchBlock[:,:,0],0,B2[:,:,0],overlapType,overlapSize)
            else:
                overlapType = 'b'
                B1 = finalImage[startX:endX,B1StartY:B1EndY+1,:]
                B2 = finalImage[B1StartX:B1EndX+1, startY:endY, :]
                # print(B1.shape,B2.shape,matchBlock.shape)
                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],B2[:,:,0],overlapType,overlapSize)
            
            maskd = np.repeat(np.expand_dims(mask,axis=2),3,axis=2)
            maskNegate = maskd==0


            finalImage[startX:endX,startY:endY,:] = maskNegate*finalImage[startX:endX,startY:endY,:]
            finalImage[startX:endX,startY:endY,:] = matchBlock*maskd+finalImage[startX:endX,startY:endY,:]
            
            finalImage_lum[startX:endX,startY:endY] = (mask==0)*finalImage_lum[startX:endX,startY:endY]
            finalImage_lum[startX:endX,startY:endY] = matchBlock_lum*mask+finalImage_lum[startX:endX,startY:endY]
            
            finalImage_grd[startX:endX,startY:endY] = (mask==0)*finalImage_grd[startX:endX,startY:endY]
            finalImage_grd[startX:endX,startY:endY] = matchBlock_grd*mask+finalImage_grd[startX:endX,startY:endY]

            completion = 100.0/noOfBlocksInRow*(i + j*1.0/noOfBlocksInCol)
            print("{0:.2f}% complete...".format(completion), end="\r", flush=True)

            if endY == outSizeY:
                break
        if endX == outSizeX:
            print("100% complete!", end="\r", flush = True)
            break
    return finalImage

scale = lambda x, top=255: (top * (x - np.min(x))) / (np.max(x) - np.min(x))
inrange = lambda x: np.where(x > 255, 255, np.where(x < 0, 0, x))
invert = lambda x: np.max(x) - x

SQDIFF = lambda sI, sG, tI, tG: np.sqrt(np.sum((sI - tI)**2)) + np.sqrt(np.sum((sG - tG)**2))

def calc_over_error(sI, sG, tI, tG, mask):

    sI, tI = sI.astype('uint8'), tI.astype('uint8')
    sG, tG = np.rint(scale(sG)).astype('uint8'), np.rint(scale(tG)).astype('uint8')

    Y, X = sI.shape
    wdth_y, wdth_x = Y//2, X//2
   
    lum = cv2.matchTemplate(tI, sI, cv2.TM_SQDIFF, mask=mask)
    grd = cv2.matchTemplate(tG, sG, cv2.TM_SQDIFF, mask=mask)

    return lum+grd

def calc_corresp_error(sI, tI, mask):

    sI, tI = sI.astype('uint8'), tI.astype('uint8')
 
    lum = cv2.matchTemplate(tI, sI, cv2.TM_SQDIFF, mask=mask)
    
    return lum

def MatchBlock(full, full_lum, full_grd, toFill, toFill_lum, toFill_grd, target_lum, target_grd, blockSize, alpha, tolerance, stochastic_mask = None):

    [m,n] = toFill.shape
 
    mask = ((toFill+0.99)>0.1).astype('uint8')
    overlap_error = np.sqrt(calc_over_error(toFill_lum, toFill_grd, full_lum, full_grd, mask))

    corresp_error = calc_corresp_error(target_lum, full_lum, mask)
    
    error = (alpha * overlap_error) + ((1-alpha) * corresp_error)

    try:
        stochastic_mask = 1-stochastic_mask
        error[stochastic_mask] = np.inf
    except:
        pass
  
    minVal = np.nanmin(error)
    
    bestBlocksx, bestBlocksy = np.where(error <= (1.0+tolerance)*minVal) ## centers of best matching patches
    
    try:
        c = np.random.randint(len(bestBlocksx))
    except:
        print(np.isnan(overlap_error).any())
        print(np.isnan(corresp_error).any())
        print(alpha, (1-alpha))
        print(np.max(overlap_error), np.max(corresp_error), (alpha * np.max(overlap_error)) + ((1-alpha) * np.max(corresp_error)))
        print(minVal)
        print(np.where(error <= minVal * (1.0+tolerance)))
        raise ValueError
        
    x, y = bestBlocksx[c], bestBlocksy[c]
    return full[x:x+m, y:y+n], full_lum[x:x+m, y:y+n], full_grd[x:x+m, y:y+n], x, y
