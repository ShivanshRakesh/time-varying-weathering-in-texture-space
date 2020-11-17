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


def Construct(textureImgArray, targetImgArray, blockSize, overlapSize, alpha=0.1, tolerance=0.1, finalImage=None):
    
    # textureImgArray = np.array(textureImgArray)
    # targetImgArray = np.array(targetImgArray)
    texture_lum, texture_grd = decouple(textureImgArray)
    target_lum, target_grd = decouple(targetImgArray)

    # print(textureImgArray.shape, targetImgArray.shape)
    outSizeX = targetImgArray.shape[0]
    outSizeY = targetImgArray.shape[1]
    [m,n,c] = textureImgArray.shape

    blocks = []
    for i in range(m-blockSize[0]):
        for j in range(n-blockSize[1]):
            #blocks are added to a list
            blocks.append(textureImgArray[i:i+blockSize[0],j:j+blockSize[1],:])                              
    blocks = np.array(blocks)
    
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

            
            # if targetBlock.shape != blocks.shape[1:]:
            #     blocks1 = []
            #     for x in range(m - targetBlock.shape[0]):
            #         for y in range(n - targetBlock.shape[1]):
            #             blocks1.append(textureImgArray[x:x+targetBlock.shape[0],y:y+targetBlock.shape[1],:]) 
            #     blocks1 = np.array(blocks1)
            #     matchBlock = MatchBlock(blocks1, toFill, targetBlock, blockSize, alpha, tolerance)
            # # print(toFill.shape, targetBlock.shape)
            # #MatchBlock returns the best suited block
            # else:
            matchBlock, matchBlock_lum, matchBlock_grd = MatchBlock(textureImgArray, texture_lum, texture_grd, 
                                                                    toFill[:,:,0], toFill_lum, toFill_grd, 
                                                                    trg_lum, trg_grd, 
                                                                    blockSize, alpha, tolerance)

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
            # print('Mask', maskNegate.shape)


            finalImage[startX:endX,startY:endY,:] = maskNegate*finalImage[startX:endX,startY:endY,:]
            finalImage[startX:endX,startY:endY,:] = matchBlock*maskd+finalImage[startX:endX,startY:endY,:]
            
            finalImage_lum[startX:endX,startY:endY] = (mask==0)*finalImage_lum[startX:endX,startY:endY]
            finalImage_lum[startX:endX,startY:endY] = matchBlock_lum*mask+finalImage_lum[startX:endX,startY:endY]
            
            finalImage_grd[startX:endX,startY:endY] = (mask==0)*finalImage_grd[startX:endX,startY:endY]
            finalImage_grd[startX:endX,startY:endY] = matchBlock_grd*mask+finalImage_grd[startX:endX,startY:endY]

            completion = 100.0/noOfBlocksInRow*(i + j*1.0/noOfBlocksInCol)
            # print("{0:.2f}% complete...".format(completion), end="\r", flush=True)

            if endY == outSizeY:
                break
        if endX == outSizeX:
            # print("100% complete!", end="\r", flush = True)
            break
    return finalImage

# def SSDError(Bi, toFill, targetBlock, alpha): 
#     [m,n,p] = toFill.shape
#     #blocks to be searched are cropped to the size of empty location
#     Bi = Bi[0:m,0:n,0:p]
#     #Locations where toFill+1 gives 0 are those where any data is not stored yet. Only those which give greater than 1 are compared for best fit.
#     # print(Bi.shape, toFill.shape, targetBlock.shape)
#     lum_Bi = np.sum(Bi, axis = 2)*1.0/3
#     lum_target = np.sum(targetBlock, axis = 2)*1.0/3
#     lum_toFill = np.sum(toFill, axis = 2)*1.0/3
#     error = alpha* np.sqrt( np.sum( ((toFill+0.99)>0.1)*(Bi - toFill)*(Bi - toFill) ) ) + (1-alpha)* np.sqrt( np.sum( ((lum_toFill+0.99)>0.1)*(lum_Bi - lum_target)*(lum_Bi - lum_target) ) )
#     return [error]

def SQDFF(lum_a, grd_a, lum_b, grd_b, mask):
    return mask * ((lum_a - lum_b)**2 + (grd_a - grd_b)**2)
    # return mask * (lum_a - lum_b) * (lum_a - lum_b)

scale = lambda x, top=255: (top * (x - np.min(x))) / (np.max(x) - np.min(x))
inrange = lambda x: np.where(x > 255, 255, np.where(x < 0, 0, x))
invert = lambda x: np.max(x) - x

def distance_map(sI, sG, tI, tG, mask):

    # M, N = tI.shape
    sI, tI = sI.astype('uint8'), tI.astype('uint8')
    sG, tG = np.rint(scale(sG)).astype('uint8'), np.rint(scale(tG)).astype('uint8')

    Y, X = sI.shape
    wdth_y, wdth_x = Y//2, X//2
    # try:
    #     if wdth_y:
    #         tI = np.vstack(([tI[0]]*wdth_y, tI))
    #         tG = np.vstack(([tG[0]]*wdth_y, tG))
    #     if Y-wdth_y-1:
    #         tI = np.vstack((tI, [tI[-1]]*(Y-wdth_y-1)))
    #         tG = np.vstack((tG, [tG[-1]]*(Y-wdth_y-1)))
        
    #     if wdth_x:
    #         tI = np.hstack((np.hstack([ tI[:,0].reshape(-1,1)]*wdth_x), tI))
    #         tG = np.hstack((np.hstack([tG[:,0].reshape(-1,1)]*wdth_x), tG))
    #     if X-wdth_x-1:
    #         tI = np.hstack((tI, np.hstack([tI[:,-1].reshape(-1,1)]*(X-wdth_x-1))))
    #         tG = np.hstack((tG, np.hstack([tG[:,-1].reshape(-1,1)]*(X-wdth_x-1))))
            
    # except Exception:
    #     print("\n", wdth_x, wdth_y, Y-wdth_y-1, X-wdth_x-1)
    #     raise Exception   

    lum = cv2.matchTemplate(tI, sI, cv2.TM_SQDIFF, mask=mask)
    grd = cv2.matchTemplate(tG, sG, cv2.TM_SQDIFF, mask=mask)

    return lum+grd

def lum_map(sI, sG, tI, tG, mask):

    # M, N = tI.shape
    sI, tI = sI.astype('uint8'), tI.astype('uint8')
    sG, tG = np.rint(scale(sG)).astype('uint8'), np.rint(scale(tG)).astype('uint8')

    Y, X = sI.shape
    wdth_y, wdth_x = Y//2, X//2
    # try:
    #     if wdth_y:
    #         tI = np.vstack(([tI[0]]*wdth_y, tI))
    #         tG = np.vstack(([tG[0]]*wdth_y, tG))
    #     if Y-wdth_y-1:
    #         tI = np.vstack((tI, [tI[-1]]*(Y-wdth_y-1)))
    #         tG = np.vstack((tG, [tG[-1]]*(Y-wdth_y-1)))
        
    #     if wdth_x:
    #         tI = np.hstack((np.hstack([ tI[:,0].reshape(-1,1)]*wdth_x), tI))
    #         tG = np.hstack((np.hstack([tG[:,0].reshape(-1,1)]*wdth_x), tG))
    #     if X-wdth_x-1:
    #         tI = np.hstack((tI, np.hstack([tI[:,-1].reshape(-1,1)]*(X-wdth_x-1))))
    #         tG = np.hstack((tG, np.hstack([tG[:,-1].reshape(-1,1)]*(X-wdth_x-1))))
            
    # except Exception:
    #     print("\n", wdth_x, wdth_y, Y-wdth_y-1, X-wdth_x-1)
    #     raise Exception   

    lum = cv2.matchTemplate(tI, sI, cv2.TM_SQDIFF, mask=mask)
    
    return lum

def MatchBlock(full, full_lum, full_grd, toFill, toFill_lum, toFill_grd, target_lum, target_grd, blockSize, alpha, tolerance):
    # error = []
    [m,n] = toFill.shape
    # bestBlocks = []
    # count = 0
    # for i in range(blocks.shape[0]):
    #     #blocks to be searched are cropped to the size of empty location
    #     Bi = blocks[i,:,:,:]
    #     Bi = Bi[0:m,0:n,0:p]
    #     error.append(SSDError(Bi, toFill, targetBlock, alpha))
    
    mask = ((toFill+0.99)>0.1).astype('uint8')
    overlap_error = np.sqrt(distance_map(toFill_lum, toFill_grd, full_lum, full_grd, mask))

    corresp_error = lum_map(target_lum, target_grd, full_lum, full_grd, mask)
    # print(np.isnan(overlap_error).any(), np.isnan(corresp_error).any())
    error = (alpha * overlap_error) + ((1-alpha) * corresp_error)
    # ((toFill+0.99)>0.1) *(cmap_toFill-cmap_target)*(cmap_toFill-cmap_target)))
    # maxVal = np.max(error)
    # error[:m//2, :] = np.inf
    # error[-(m//2):, :] = np.inf
    # error[:, :n//2] = np.inf
    # error[:, -(n//2):] = np.inf
    # error = np.where(np.all([bestBlocksx >= m//2, bestBlocksx < m-m//2, bestBlocksy >= n//2, bestBlocksy < n-n//2], axis=0), error, maxVal)
    minVal = np.nanmin(error)
    # bestBlocks = [block[:m, :n, :p] for i, block in enumerate(blocks) if error[i] <= (1.0+tolerance)*minVal]
   
    bestBlocksx, bestBlocksy = np.where(error <= (1.0+tolerance)*minVal) ## centers of best matching patches
    # print(bestBlocksx.shape)
    # valid = np.where(np.all([bestBlocksx >= m//2, bestBlocksx < m-m//2, bestBlocksy >= n//2, bestBlocksy < n-n//2], axis=0))
    # bestBlocksx, bestBlocksy = bestBlocksx[valid], bestBlocksy[valid]
    try:
        c = np.random.randint(len(bestBlocksx))
    except:
        print(np.isnan(overlap_error).any())
        print(np.isnan(corresp_error).any())
        print(minVal)
        plt.imshow(toFill_lum, cmap='gray')
        plt.show()
        plt.imshow(error, cmap='gray')
        plt.show()
        raise ValueError
    x, y = bestBlocksx[c], bestBlocksy[c]
    # print(x-m//2,':',x+m-m//2,'\t', y-n//2,':', y+n-n//2, '\tMin val', minVal, '\tGottenVal', error[x,y])
    # return full[x-m//2:x+m-m//2, y-n//2:y+n-n//2], full_lum[x-m//2:x+m-m//2, y-n//2:y+n-n//2], full_grd[x-m//2:x+m-m//2, y-n//2:y+n-n//2]
    # print(x,':',x+m,'\t', y,':', y+n, '\tMin val', minVal, '\tGottenVal', error[x,y])
    return full[x:x+m, y:y+n], full_lum[x:x+m, y:y+n], full_grd[x:x+m, y:y+n]
   
    # for i in range(blocks.shape[0]):
    #     if error[i] <= (1.0+tolerance)*minVal:
    #         block = blocks[i,:,:,:]
    #         bestBlocks.append(block[0:m,0:n,0:p])
    #         count = count+1
    # c = np.random.randint(len(bestBlocks))
    # return bestBlocks[c]
    
    
    # [minError,bestBlock] = SSDError(blocks[0,:,:,:], toFill, targetBlock, alpha)
    # for i in range(blocks.shape[0]):
    #     [error,Bi] = SSDError(blocks[i,:,:,:], toFill, targetBlock, alpha)
    #     if minError > error:
    #         bestBlock = Bi
    #         minError = error
    # return bestBlock

# def LoadImage( infilename ) :
#     img = Image.open(infilename).convert('RGB')
#     data = np.asarray(img)
#     return data

# def SaveImage( npdata, outfilename ) :
#     print(npdata.shape)
#     img = Image.fromarray(npdata.astype('uint8')).convert('RGB')
#     img.save( outfilename )

# texture = np.array(LoadImage('../images/scribble.jpg'))
# target = np.array(LoadImage('../images/tendulkar.jpg'))

# # print(data.shape)
# out = Construct(texture, target, [15, 15], 5, 0.1, 0.1)
# SaveImage(out,'../results/transfer/scribble_tendulkar_b15_o5_a0_1.png')