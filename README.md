# Image stitching using SIFT
 simple stitchin project which use SIFT as Keypoint for detecting and describtion then use RANSAK for Estimating the Homography and distance transform for Blending
Simple project for image stitching as follow: 

1- Registration (point-to-point matching): 

Detect and describe keypoints using SIFT.

Find a point to point mapping using KNN

Sort the matches to find best matches

2- Reprojection to single plane: 

Estimate the homography matrix using RANSAC. 

"Normalize" the homography matrix to produce positive coordinates (multiplying by transition matrix)

3- blinding ( removing the artifacts): by using distance transform.

 
 # Results
![output](https://github.com/zaky-fetoh/stitching/blob/main/out.jpg)
