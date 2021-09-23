# https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/
from __future__ import print_function
import cv2 as cv
import numpy as np
from random import randint
import matplotlib.pyplot as plt    
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.5
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
    cv.imwrite('canvas.jpg',canvas)

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
def alignImages(im1, im2,band=None):
    
    # Convert images to grayscale
    im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
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
    # imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches,None)
    imMatches,displacements = my_draw_matches(im1, keypoints1, im2, keypoints2, matches,None)


    cv.imwrite('matches'+'.jpg', imMatches)
    hist=np.histogram(displacements[:,0])
    x_displacement=hist[1][np.argmax(hist[0])]
    plt.clf() 
    plt.xlim(-600,600) #! cheat
    plt.hist(displacements[:,0],bins=100)
    plt.savefig('original hist.png')
    print('[+] x displacement is {:.2f}'.format(x_displacement))
    # print('[+] x displacement range [{:.2f} , {:.2f}]'.format(x_displacement+band[0],x_displacement+band[1]))
    
    if not band is None:    
        filtered_matches=[]
        for i in range(len(matches)):
            if band[0]<matches[i].distance<band[1]:
                filtered_matches.append(matches[i])
        imMatches,displacements = my_draw_matches(im1, keypoints1, im2, keypoints2, filtered_matches,None)

        cv.imwrite('matches filtered'+'.jpg', imMatches)
        hist=np.histogram(displacements[:,0])
        x_displacement=hist[1][np.argmax(hist[0])]
        plt.clf() 
        plt.xlim(-600,600) #! cheat
        plt.hist(displacements[:,0],bins=100)
        plt.savefig('filtered hist.png')
    print('[+] filtered x displacement within bands of {} is {:.2f}'.format(band,x_displacement))

    

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

    return im1Reg,im1Reg_filtered, h,h_filtered
# -----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Read reference image
    # refFilename = "form.jpg"
    # refFilename = "original form.jpg"
    # refFilename = "g2.jpg"
    refFilename = "im1.jpg"



    imReference = cv.imread(refFilename, cv.IMREAD_COLOR)
    
    # Read image to be aligned
    # imFilename = "scanned-form.jpg"
    # imFilename = "rotated.jpg"
    # imFilename = "g1.jpg"
    imFilename = "im2.jpg"



    im = cv.imread(imFilename, cv.IMREAD_COLOR)
    
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg,imReg_filtered, h, h_filtered = alignImages(im, imReference,[0,20])

    # Write aligned image to disk.
    cv.imwrite("aligned.jpg", imReg)
    cv.imwrite("aligned_filtered.jpg", imReg_filtered)
    
    # Print estimated homography