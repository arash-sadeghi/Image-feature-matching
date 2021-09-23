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
BAND=[-30,30]
distance_thresh=60
# -----------------------------------------------------------------------------------------------
def calculate_homography(im1, keypoints1, im2, keypoints2, matches):
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
def draw_results(im1, keypoints1, im2, keypoints2, matches,method):
    imMatches,displacements = my_draw_matches(im1, keypoints1, im2, keypoints2, matches,None)
    cv.imwrite(RESULTS_FOLDER+'/'+method+'_matches'+'.jpg', imMatches)
    hist_my=np.histogram(displacements[:,0])
    x_displacement=hist_my[1][np.argmax(hist_my[0])]
    plt.clf() 
    plt.xlim(-600,600) #! cheat
    plt.hist(displacements[:,0],bins=100)
    plt.savefig(RESULTS_FOLDER+'/'+method+'_hist.png')
    print('[+] method {} x displacement is {:.2f}'.format(method,x_displacement))
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
    canvas=np.zeros((im1.shape[0]+im2.shape[0],max(im1.shape[1],im2.shape[1]),3))
    canvas[0:im1.shape[0],0:im1.shape[1]]=im1
    canvas[im1.shape[0]:im1.shape[0]+im2.shape[0],0:im2.shape[1]]=im2
    displacements=np.zeros((len(keypoints1),2))
    # cv.circle(canvas,(200,200),30,(0,255,0),-1) #! for debug purpose
    for i in range(len(matches)):
        p1=(int( keypoints1[matches[i].queryIdx].pt[0] ),int(keypoints1[matches[i].queryIdx].pt[1]))
        p2=(int( keypoints2[matches[i].trainIdx].pt[0] ),int(keypoints2[matches[i].trainIdx].pt[1]+im2.shape[0]))
        displacements[i,0]=p2[0]-p1[0]
        displacements[i,1]=p2[1]-p1[1]
        color=random_color()
        cv.circle(canvas,p1,10,color,-1)
        cv.circle(canvas,p2,10,color,-1)
        cv.line(canvas, p1, p2, color, thickness=5)

    return canvas,displacements
# -----------------------------------------------------------------------------------------------
def pre_filter_match(keypoints1, descriptors1, keypoints2 ,descriptors2,matcher,band):
    #! time consuming part
    #! mathcer only acts on uint8 data type zz=np.zeros((5, len(descriptors1[c]) ),dtype=np.uint8)
    match_obj=matcher.match(descriptors1,descriptors2,None)
    counter=0
    for c,v in enumerate(keypoints1):
        nominee=[]
        nominee_indx=[]
        for cc,vv in enumerate(keypoints2):
            if band[0]<=v.pt[0]-vv.pt[0]<=band[1]:
                # arg1=np.zeros((1,))
                nominee.append(descriptors2[cc])
                nominee_indx.append(cc)
        if len(nominee)==0: # no match found
            continue
        nominee=np.array(nominee)
        tricker=np.zeros( nominee.shape , dtype=np.uint8 )
        tricker[:]=descriptors1[c]
        #! [0] below is because all trainIdx are the same
        # matches_indx=matcher.match(tricker,nominee,None)[0]
        matched=matcher.match(tricker,nominee,None)[0]
        if matched.distance>=distance_thresh:
            continue
        else:
            matches_indx=matched.trainIdx
        match_obj[counter].queryIdx=c
        match_obj[counter].trainIdx=nominee_indx[matches_indx]
        counter+=1

    return match_obj[0:counter+1]
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
    #! ORB only creates the features. Matching is different process

    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    if method=='opencv' or method=='post_filter':
        # Match features.
        matches = matcher.match(descriptors1, descriptors2, None)

    elif method=='pre_filter':
        matches = pre_filter_match(keypoints1, descriptors1, keypoints2 ,descriptors2,matcher,band)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]
    distances=np.array([matches[_].distance for _ in range(len(matches))])
    matches=matches[0:np.where(distances>distance_thresh)[0][0]]    


    # Draw top matches
    # imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches,None)
    # cv.imwrite(RESULTS_FOLDER+'/'+'opencv_matches'+'.jpg', imMatches)


    if method=='post_filter':
        #! post filtering
        if band is None:    
            raise NameError('[-] no band given for post_filter')

        filtered_matches=[]
        for i in range(len(matches)):
            #! norm2 distance
            # dist=np.linalg.norm(np.array(keypoints1[matches[i].queryIdx].pt)-np.array(keypoints2[matches[i].trainIdx].pt))
            
            #! 1D distance
            dist=keypoints1[matches[i].queryIdx].pt[0]-keypoints2[matches[i].trainIdx].pt[0]
            #!? why some keypoint locations has floating parts arent they coordiantes
            if band[0]<=dist<=band[1]:
                filtered_matches.append(matches[i])
        draw_results(im1, keypoints1, im2, keypoints2, filtered_matches,method)
        # im1Reg_filtered, h_filtered=calculate_homography(im1, keypoints1, im2, keypoints2, filtered_matches)
        # return im1Reg_filtered, h_filtered
        return None,None
    elif method=='pre_filter':
        draw_results(im1, keypoints1, im2, keypoints2, matches,method)
        # im1Reg, h=calculate_homography(im1, keypoints1, im2, keypoints2, matches)
        # return im1Reg, h
        return None,None
    elif method=='opencv':
        # draw_results(im1, keypoints1, im2, keypoints2, matches,method)
        imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches,None)
        cv.imwrite(RESULTS_FOLDER+'/'+'opencv_matches'+'.jpg', imMatches)
        # im1Reg, h=calculate_homography(im1, keypoints1, im2, keypoints2, matches)
        # return im1Reg, h
        return None,None

# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # creat results folder
    if not ( RESULTS_FOLDER in os.listdir() ):
        os.mkdir(RESULTS_FOLDER)

    
    all_ims,all_files=read_dataset()

    im1=all_ims[0,0,:,:,:]
    
    im2=all_ims[1,0,:,:,:]

    truth=all_files[1,0,:]

    """Registered image will be resotred in imReg."""
    """The estimated homography will be stored in h."""
    
    imReg, h = alignImages(im1,im2,method='opencv')
    
    imReg_post_filtered, h_post_filtered = alignImages(im1,im2,band=BAND,method='post_filter')

    imReg_post_filtered, h_post_filtered = alignImages(im1,im2,band=BAND,method='pre_filter')


    print('[+] Truth is {}'.format(truth))
    # Write aligned image to disk.
    # cv.imwrite(RESULTS_FOLDER+'/'+"aligned.jpg", imReg)
    # cv.imwrite(RESULTS_FOLDER+'/'+"aligned_filtered.jpg", imReg_post_filtered)
    
    # Print estimated homography