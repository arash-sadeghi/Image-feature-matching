# https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/
from __future__ import print_function
import cv2 as cv
import numpy as np
from random import randint
import matplotlib.pyplot as plt    
import os
from time import ctime,time
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.5
RESULTS_FOLDER = 'results '+ctime(time()).replace(':','_')
# -----------------------------------------------------------------------------------------------
def read_dataset(location=None,dataset_folder_name='GRIEF-datasets',subfolder='michigan',im_extention='bmp'):
    if location is None and subfolder is None:
        location=os.getcwd()
        print('[+] dataset address not given. DEFAULT: Looking in current dictory {} for subfolder {}'\
            .format(os.getcwd(),subfolder))
    
    folders=os.listdir(dataset_folder_name+'/'+subfolder)
    folders.sort()
    all_files=[]
    all_images=[]


    for folder in folders:
        # if that folder is not empty
        if len( os.listdir(dataset_folder_name+'/'+subfolder+'/'+folder) ) > 0:
            # start to read the images
            ims=[]
            files=[]
            for im in os.listdir(dataset_folder_name+'/'+subfolder+'/'+folder):
                
                if os.path.splitext(dataset_folder_name+'/'+subfolder+'/'+folder+'/'+im)[1][1:]\
                ==im_extention:
                    # if file is image appedn it to list of images
                    ims.append(cv.imread(dataset_folder_name+'/'+subfolder+'/'+folder+'/'+im,\
                    cv.IMREAD_COLOR))
                
                else:
                    # if file is not image appedn it to list of files
                    with open(dataset_folder_name+'/'+subfolder+'/'+folder+'/'+im,'r') as file:
                        # files.append(file.read())
                        files.append(np.loadtxt(file))

            all_files.append(files)            
            all_images.append(ims)
            ims=[]                            
            files=[]

    return np.array(all_images),np.array(all_files).squeeze()
# -----------------------------------------------------------------------------------------------
def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))
# -----------------------------------------------------------------------------------------------
def my_draw_matches(im1, keypoints1, im2, keypoints2, matches,_):    
    # canvas=np.zeros((max(im1.shape[0],im2.shape[0]),im1.shape[1]+im2.shape[1],3))
    # canvas[0:im1.shape[0],0:im1.shape[1]]=im1
    # canvas[0:im2.shape[0] , im1.shape[1]:im1.shape[1]+im2.shape[1]]=im2

    canvas=np.zeros((im1.shape[0]+im2.shape[0],max(im1.shape[1],im2.shape[1]),3))
    canvas[0:im1.shape[0],0:im1.shape[1]]=im1
    canvas[im1.shape[0]:im1.shape[0]+im2.shape[0],0:im2.shape[1]]=im2
    # cv.imshow('canvas',canvas)
    # cv.waitKey()
    cv.imwrite(RESULTS_FOLDER+'/'+'canvas.jpg',canvas)

    displacements=np.zeros((len(keypoints1),2))
    # for i in range(len(keypoints1)):
    #     p1=(int(keypoints1[i].pt[0]),int(keypoints1[i].pt[1]))
    #     # p2=(int(keypoints2[i].pt[0]+im2.shape[1]),int(keypoints2[i].pt[1]))
    #     p2=(int(keypoints2[i].pt[0]),int(keypoints2[i].pt[1]+im2.shape[0]))
    #     displacements[i,0]=p2[0]-p1[0]
    #     displacements[i,1]=p2[1]-p1[1]
    #     color=random_color()
    #     cv.circle(canvas,p1,10,color,-1)
    #     cv.circle(canvas,p2,10,color,-1)
    #     cv.line(canvas, p1, p2, color, thickness=5)

    for i in range(len(matches)):
        p1=(int(keypoints1[matches[i].queryIdx].pt[0]),int(keypoints1[matches[i].queryIdx].pt[1]))
        p2=(int(keypoints2[matches[i].trainIdx].pt[0]),int(keypoints2[matches[i].trainIdx].pt[1]+im2.shape[0]))
        displacements[i,0]=p2[0]-p1[0]
        displacements[i,1]=p2[1]-p1[1]
        color=random_color()
        cv.circle(canvas,p1,10,color,-1)
        cv.circle(canvas,p2,10,color,-1)
        cv.line(canvas, p1, p2, color, thickness=5)

    return canvas,displacements
# -----------------------------------------------------------------------------------------------
def alignImages(im1, im2,band=None,method='opencv'):
    
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
    #! sift and *super pixel* 
    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    # Match features.
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]
    
    # Draw top matches
    imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches,None)
    cv.imwrite(RESULTS_FOLDER+'/'+'opencv_matches'+'.jpg', imMatches)
    
    if method=='mine':
        imMatches_my,displacements_my = my_draw_matches(im1, keypoints1, im2, keypoints2, matches,None)
        cv.imwrite(RESULTS_FOLDER+'/'+'my_matches'+'.jpg', imMatches_my)
        hist_my=np.histogram(displacements_my[:,0])
        x_displacement=hist_my[1][np.argmax(hist_my[0])]
        plt.clf() 
        plt.xlim(-600,600) #! cheat
        plt.hist(displacements_my[:,0],bins=100)
        plt.savefig(RESULTS_FOLDER+'/'+'my_original_hist.png')
        print('[+] my x displacement is {:.2f}'.format(x_displacement))
        # print('[+] x displacement range [{:.2f} , {:.2f}]'.format(x_displacement+band[0],x_displacement+band[1]))
        
        #! post filtering
        if not band is None:    
            filtered_matches=[]
            for i in range(len(matches)):
                if band[0]<matches[i].distance<band[1]:
                    filtered_matches.append(matches[i])
            imMatches_filtered,displacements_filtered = my_draw_matches(im1, keypoints1, im2, keypoints2, filtered_matches,None)

            cv.imwrite(RESULTS_FOLDER+'/'+'matches_post_filtered'+'.jpg', imMatches_filtered)
            hist_filtered=np.histogram(displacements_filtered[:,0])
            x_displacement=hist_filtered[1][np.argmax(hist_filtered[0])]
            plt.clf() 
            plt.xlim(-600,600) #! cheat
            plt.hist(displacements_filtered[:,0],bins=100)
            plt.savefig(RESULTS_FOLDER+'/'+'post_filtered_hist.png')
        # Extract location of good matches --------------------------------------
        points1 = np.zeros((len(filtered_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(filtered_matches), 2), dtype=np.float32)
        
        for i, match in enumerate(filtered_matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        
        # Find homography
        h_filtered, mask = cv.findHomography(points1, points2, cv.RANSAC)
        
        # Use homography
        height, width, channels = im2.shape
        im1Reg_filtered = cv.warpPerspective(im1, h_filtered, (width, height))

        print('[+] post filtered x displacement within bands of {} is {:.2f}'.format(band,x_displacement))
    
        return im1Reg_filtered, h_filtered

    elif method=='opencv':
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        
        # Find homography
        h, mask = cv.findHomography(points1, points2, cv.RANSAC)
        
        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv.warpPerspective(im1, h, (width, height))
        

        return im1Reg, h
# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # creat results folder
    if not ( RESULTS_FOLDER in os.listdir() ):
        os.mkdir(RESULTS_FOLDER)

    
    all_ims,all_files=read_dataset()

    im1=all_ims[0,0,:,:,:]
    
    im2=all_ims[1,0,:,:,:]

    truth=all_files[1,0,:]

    print('ji')

    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im1,im2,method='opencv')
    imReg_post_filtered, h_post_filtered = alignImages(im1,im2,band=[-100,100],method='mine')

    print('[+] Truth is {}'.format(truth))
    # Write aligned image to disk.
    cv.imwrite(RESULTS_FOLDER+'/'+"aligned.jpg", imReg)
    cv.imwrite(RESULTS_FOLDER+'/'+"aligned_filtered.jpg", imReg_post_filtered)
    
    # Print estimated homography