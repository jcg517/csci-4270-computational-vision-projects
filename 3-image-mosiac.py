import numpy as np
import cv2
from matplotlib import pyplot as plt
from itertools import combinations
import sys
import os

in_dir = sys.argv[1]
out_dir = sys.argv[2]

# Declare parameters
MATCH_FRAC_THRESHOLD = 0.03
DIST_RATIO_THRESHOLD = 0.7
F_INLIER_COUNT_THRESHOLD = 10
F_INLIER_PERCENT_THRESHOLD = 70
BLENDING_FACTOR = .5

# Declare list variables. 
images = []					# Image pixel data by image index
images_color = []			# RGB pixel data by image index
fnames = []					# Image file names by image index
keypoints = []				# Sift keypoints by image index
descriptors = []			# SIFT descriptors by image index
match_pairs = []			# Image indice tuples pairs, representing matches: (i, j); is narrowed down as matches are removed
matched_keypoints = []		# All the corresponding keypoints of good matches, indexed by image num
all_good_matches = {}		# All good matches indexed by image match pair: (i, j)
F_inlier_match_masks = {}	# Inlier match mask data consistent with fundemental matrix, indexed by (i, j)
H_inlier_match_masks = {}	# Inlier match mask data consistent with homography matrix, indexed by (i, j)
F_matrices = {}				# Fundamental matricies between images i,j to form mosaic from, indexed by (i, j)
H_matrices = {}				# Homography matricies between images i,j to form mosaic from, indexed by (i, j)


# Read Images
for fn in os.listdir(in_dir):
	fnames.append(fn)
	images.append(cv2.imread(f"{in_dir}\{fn}", cv2.IMREAD_GRAYSCALE))
	images_color.append(cv2.imread(f"{in_dir}\{fn}", cv2.IMREAD_COLOR))


# Find keypoints and descriptors using SIFT
sift = cv2.SIFT_create()
for idx, im in enumerate(images):
    kp, des = sift.detectAndCompute(im.astype(np.uint8),None)
    keypoints.append(kp)
    descriptors.append(des)
    sift_out_img = cv2.drawKeypoints(im.astype(np.uint8), kp, None)   

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(sift_out_img, cv2.COLOR_BGR2RGB))  
    plt.title(f"Keypoints of {fnames[idx]}")
    plt.axis('off')
    plt.show()

    print(f"Keypoints detected in {fnames[idx]}: {len(kp)}\n")

   
# Draw and determine good matches
for (i, j) in combinations(range(0, len(images)), 2):
    des_i, des_j = descriptors[i], descriptors[j]
    kps_i, kps_j = keypoints[i], keypoints[j]
    img_i, img_j = images[i], images[j]

    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des_i, des_j, k=2)
    matched_kps_i, matched_kps_j = [], [] 

    # ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < DIST_RATIO_THRESHOLD * n.distance:
            good_matches.append(m)
            matched_kps_i.append(kps_i[m.queryIdx].pt)
            matched_kps_j.append(kps_j[m.trainIdx].pt) 

    out_img = cv2.drawMatches(img_i, kps_i, img_j, kps_j, good_matches, None, flags=2)
    match_frac_i = round(len(good_matches) / len(kps_i), 4)
    match_frac_j = round(len(good_matches) / len(kps_j), 4)

    print(f"Pair: {fnames[i]}, {fnames[j]}:")
    print(f"Match fraction for {fnames[i]}: {match_frac_i}")
    print(f"Match fraction for {fnames[j]}: {match_frac_j}")

    if match_frac_i < MATCH_FRAC_THRESHOLD and match_frac_j < MATCH_FRAC_THRESHOLD:
        print(f"Since both matching fractions fall BELOW the threshold of {MATCH_FRAC_THRESHOLD}, " 
              f"the images DO NOT match and show DIFFERENT scenes.\n")
    else:
        print(f"Since either matching fractions fall ABOVE the threshold of {MATCH_FRAC_THRESHOLD},"
               f"the images DO  match and show THE SAME scenes.\n")
        match_pairs.append((i, j))
        matched_keypoints.append((np.array(matched_kps_i, dtype=np.float32), 
                                  np.array(matched_kps_j, dtype=np.float32)))
        all_good_matches[(i, j)] = good_matches
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))  
    plt.title(f"Good Matches Between {fnames[i]} and {fnames[j]}")
    plt.axis('off')
    plt.show()

# Find Inlier Matches using Fundemental Matrix Estimation
for idx, (i, j) in enumerate(match_pairs.copy()):
    pts_i, pts_j = matched_keypoints[idx]
    kps_i, kps_j = keypoints[i], keypoints[j]
    img_i, img_j = images[i], images[j]

    fundamental_mat, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 
                                                   ransacReprojThreshold=3, confidence=.99)
    F_inlier_matches_mask = mask.ravel().tolist()
    F_inlier_matches_count = np.sum(F_inlier_matches_mask)
    F_inlier_match_percent = (F_inlier_matches_count / len(F_inlier_matches_mask)) * 100

    F_inlier_match_masks[(i, j)] = np.array(F_inlier_matches_mask) # Store match data for later use
    F_matrices[(i, j)] = fundamental_mat

    print(f"Pair: {fnames[i]}, {fnames[j]}:")
    print(f"Number of F-matrix inlier matches: {F_inlier_matches_count}")
    print(f"Percentage of F-matrix inlier matches: {F_inlier_match_percent:.2f}%\n")
    
    # Determine if image i and j match based on percent and number of inlier matches
    if F_inlier_matches_count < F_INLIER_COUNT_THRESHOLD:
        print(f"Since cound of F-matrix inlier matches falls BELOW the threshold of " 
              f"{F_INLIER_COUNT_THRESHOLD}%, the images CANNOT be matched.\n")
        # remove disqualified matches 
        match_pairs.remove((i, j))
        
    elif F_inlier_match_percent >= F_INLIER_PERCENT_THRESHOLD:
        print(f"Since percentage of F-matrix inlier matches falls ABOVE the threshold of "
              f"{F_INLIER_PERCENT_THRESHOLD}%, the images DO match and show the SAME scenes.\n")
       
    else:
        print(f"Since percentage of F-matrix inlier matches falls BELOW th e threshold of " 
              f"{F_INLIER_PERCENT_THRESHOLD}%, the images do NOT match and show DIFFERENT scenes.\n")
        match_pairs.remove((i, j))
        
    # Draw and plot the image    
    out_img = cv2.drawMatches(img_i, kps_i, img_j, kps_j, all_good_matches[(i, j)], None, 
                              matchesMask=F_inlier_matches_mask, flags=2)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))  
    plt.title(f"F-matrix Inlier Matches Between: {fnames[i]} and {fnames[j]} ")
    plt.axis('off')
    plt.show()

    # Compute epipolar lines
    epi_lines = cv2.computeCorrespondEpilines(pts_j.astype(np.int32), 1, fundamental_mat,None)
    epi_img = cv2.cvtColor(img_j.copy(), cv2.COLOR_BGR2RGB)
    for [line] in epi_lines:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        epi_img = cv2.line(epi_img, (int(-line[2] / line[0]),0), (0, int(-line[2] / line[1])), color)
        
    plt.title(f"Epipolar lines for img-j ")
    plt.imshow(epi_img)
    plt.axis('off')

# Calculate and Compare with Homography Estimate Matrix
for idx, (i, j) in enumerate(match_pairs.copy()):
    pts_i, pts_j = matched_keypoints[idx]
    kps_i, kps_j = keypoints[i], keypoints[j]
    img_i, img_j = images[i], images[j]

    homography_mat, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 5.0)
    H_inlier_matches_mask = mask.ravel().tolist()
    H_inlier_matches_count = np.sum(H_inlier_matches_mask)
    H_inlier_match_percent = (H_inlier_matches_count / len(H_inlier_matches_mask)) * 100
    H_matrices[(i, j)] = homography_mat
    H_inlier_match_masks = np.array(H_inlier_matches_mask)
    
    F_H_inlier_diffs_count = np.sum(F_inlier_match_masks[(i, j)] ^ np.array(H_inlier_matches_mask))
    F_H_inlier_diffs_ratio = F_H_inlier_diffs_count / len(H_inlier_matches_mask)

    print(f"H-Matrix Inlier Matches Between: {fnames[i]} and {fnames[j]}")
    print(f"Number of H-Matrix Inlier Matches: {H_inlier_matches_count}")
    print(f"Percentage of H-Matrix Inlier Matches: {H_inlier_match_percent:.2f}%")
    print(f"Percentage inlier matches shared between matrix estimates: {(1 - F_H_inlier_diffs_ratio)*100:.2f}%")

    # Decide if mosiac can be made if most inlier matches from F matrix also kept as inliers in H
    if F_H_inlier_diffs_ratio > .5:
        print(f"Since most inlier matches from the F-matrix estimate are NOT kept as inlier matches "
              f"to the H-matrix, the images CANNOT be accurately aligned.\n")
        # Remove the pairs from list of matches if disqualified
        match_pairs.remove((i, j))
        
    else:
        print(f"Since most inlier matches from the F-matrix estimate ARE kept as inlier matches to the"
              f"H-matrix, the images CAN be accurately aligned.\n")

    # Draw and plot the images
    out_img = cv2.drawMatches(img_i,kps_i,img_j,kps_j, all_good_matches[(i, j)],None, 
                              matchesMask=H_inlier_matches_mask, flags=2)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    plt.title(f"H-Matrix Inlier Matches Between: {fnames[i]} and {fnames[j]}")
    plt.axis('off')
    plt.show()


# Mapping the images
if not match_pairs:
    print("There are no viable images to match")
    
for (i, j) in match_pairs:
    img_i, img_j = images_color[i], images_color[j]
    m, n = img_j.shape[0], img_j.shape[1]       # input img dimesions
    M, N = img_j.shape[0]*2, img_j.shape[1]*2   # new canvas dimensions twice the size
    
    H_inv = np.linalg.inv(H_matrices[(i, j)])

    # Create canvas and place the first image, and warped second image
    canvas = np.zeros((M, N, 3), dtype=np.uint8)
    canvas[0:m, 0:n] = img_i
    warped = cv2.warpPerspective(img_j, H_inv, (N, M)) # apply homography

    #overlaying warped imgj onto the canvas
    canvas = np.where(warped > 0, warped, canvas)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))  
    plt.title(f"Mosiac of {fnames[i]} and {fnames[j]}:")
    plt.axis('off')
    plt.show()

    # Save file
    # extension = fnames[i][-4:]
    # cv2.imwrite(f"{out_dir}\{fnames[i][:-4]}_{fnames[j][:-4]}{extension}", canvas)
