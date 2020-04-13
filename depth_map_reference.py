# website: https://vgg.fiit.stuba.sk/2015-02/2783/
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration


def calibrate_camera(img_points, obj_points, image_size):
    '''
    Calibration is carried out on all images
    :return:
    '''

    # camera matrix, distortion coefficients, rotation and translation vectors etc
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    # np.savez("calibration_output.npz", ret=ret, mtx=mtx, rvecs=rvecs, tvecs=tvecs, dist=dist)
    return ret, mtx, dist, rvecs, tvecs


def undistort_image(mtx, dist, w, h):
    i = 0
    # refind camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    for fname in images:
        img = cv2.imread(fname)

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imwrite('./calibration/output/calibresult{}.png'.format(i+1), dst)

        i += 1


def stero_calibration(calib_files_left, calib_files_right):
    img_left_points = []
    img_right_points = []
    obj_points = []
    # pattern_size = (6, 9)
    # pattern_points = np.zeros((9*6,3), np.float32)
    # pattern_points[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    pattern_size = (9, 6)
    pattern_points = np.zeros((9*6,3), np.float32)
    pattern_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # print(pattern_points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size = 0
    calib_files = zip(calib_files_left, calib_files_right)

    for left_path, right_path in calib_files:
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        gray_left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        image_size = gray_left_img.shape
        # print(image_size)

        find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

        left_found, left_corners = cv2.findChessboardCorners(gray_left_img, pattern_size, flags=find_chessboard_flags)
        right_found, right_corners = cv2.findChessboardCorners(gray_right_img, pattern_size, flags=find_chessboard_flags)

        if left_found:
            cv2.cornerSubPix(gray_left_img, left_corners, (11, 11), (-1, -1), criteria)
        if right_found:
            cv2.cornerSubPix(gray_right_img, right_corners, (11, 11), (-1, -1), criteria)

        if left_found and right_found:
            img_left_points.append(left_corners)
            img_right_points.append(right_corners)
            obj_points.append(pattern_points)

        cv2.imshow("left", left_img)
        cv2.drawChessboardCorners(left_img, pattern_size, left_corners, left_found)
        cv2.drawChessboardCorners(right_img, pattern_size, right_corners, right_found)

        cv2.imshow("left chess", left_img)
        cv2.imshow("right chess", right_img)
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = calibrate_camera(img_left_points, obj_points, image_size)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = calibrate_camera(img_right_points, obj_points, image_size)


    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereocalib_flag = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    # stereocalib_flag = cv2.CALIB_FIX_INTRINSIC#cv2.CALIB_USE_INTRINSIC_GUESS #cv2.CALIB_FIX_INTRINSIC
    stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objectPoints= obj_points,
        imagePoints1=img_left_points,
        imagePoints2=img_right_points,
        cameraMatrix1=mtx_l,
        distCoeffs1=dist_l,
        cameraMatrix2=mtx_r,
        distCoeffs2=dist_r,
        imageSize=image_size,
        criteria=stereocalib_criteria,
        flags=stereocalib_flag)

    # data["stereocalib_retval"], data["cameraMatrix1"], data["distCoeffs1"], \
    # data["cameraMatrix2"], data["distCoeffs2"], data["R"], data["T"], data["E"], data["F"] = \
    #     cv2.stereoCalibrate(objectPoints=obj_points,
    #                          imagePoints1=img_left_points,
    #                          imagePoints2=img_right_points,
    #                          criteria=stereocalib_criteria,
    #                          flags=stereocalib_flags,
    #                          imageSize=image_size)
    data = {
        "stereocalib_retval": stereocalib_retval,
        "cameraMatrix1": cameraMatrix1,
        "distCoeffs1": distCoeffs1,
        "cameraMatrix2": cameraMatrix2,
        "distCoeffs2": distCoeffs2,
        "R": R,
        "T": T,
        "E": E,
        "F": F
    }
    return data


# # def stereo_rectify_uncalibrated():
# #     cv2.stereoRectifyUncalibrated(points1=, points2=, F=, imgSize=)
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        # x0, y0 = map(int, [0, r[2] / r[1]])
        # x1, y1 = map(int, [c, (r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
#
#
# def getCorners(images, chessboard_size, show=True):
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
#     objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)*3.88 # multiply by 3.88 for large chessboard squares
#
#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.
#
#     for image in images:
#         frame = cv2.imread(image)
#         # height, width, channels = frame.shape # get image parameters
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)   # Find the chess board corners
#         if ret:                                                                         # if corners were found
#             objpoints.append(objp)
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)    # refine corners
#             imgpoints.append(corners2)                                                  # add to corner array
#
#             if show:
#                 # Draw and display the corners
#                 frame = cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
#                 cv2.imshow('frame', frame)
#                 cv2.waitKey(100)
#
#     cv2.destroyAllWindows()             # close open windows
#     return objpoints, imgpoints, gray.shape[::-1]
#
# # draw the provided points on the image
# def drawPoints(img, pts, colors):
#     for pt, color in zip(pts, colors):
#         cv2.circle(img, tuple(pt[0]), 5, color, -1)
#
# # draw the provided lines on the image
# def drawLines(img, lines, colors):
#     _, c, _ = img.shape
#     for r, color in zip(lines, colors):
#         x0, y0 = map(int, [0, -r[2]/r[1]])
#         x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
#         cv2.line(img, (x0, y0), (x1, y1), color, 1)
#
# def draw_epipolar_line(imgL, imgR):
#     # use get corners to get the new image locations of the checcboard corners (undistort will have moved them a little)
#     chessboard_size = (6, 9)
#     imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
#     imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
#     colors = [255, 0, 0, 0, 255, 0]
#     _, imgpointsL, _ = getCorners(["2L34_undistorted.bmp"], chessboard_size, show=False)
#     _, imgpointsR, _ = getCorners(["2R34_undistorted.bmp"], chessboard_size, show=False)
#
#     # get 3 image points of interest from each image and draw them
#     ptsL = np.asarray([imgpointsL[0][0], imgpointsL[0][10], imgpointsL[0][20]])
#     ptsR = np.asarray([imgpointsR[0][5], imgpointsR[0][15], imgpointsR[0][25]])
#     drawPoints(imgL, ptsL, colors[3:6])
#     drawPoints(imgR, ptsR, colors[0:3])
#
#     # find epilines corresponding to points in right image and draw them on the left image
#     epilinesR = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F)
#     epilinesR = epilinesR.reshape(-1, 3)
#     drawLines(imgL, epilinesR, colors[0:3])
#
#     # find epilines corresponding to points in left image and draw them on the right image
#     epilinesL = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
#     epilinesL = epilinesL.reshape(-1, 3)
#     drawLines(imgR, epilinesL, colors[3:6])
#
#     # combine the corresponding images into one and display them
#     combineSideBySide(imgL, imgR, "epipolar_lines", save=True)

def get_epipolar_line(imgL, imgR):
    img1 = imgL
    img2 = imgR
    # img1 = cv2.imread(imgL,0)  #queryimage # left image
    # img2 = cv2.imread(imgR,0) #trainimage # right image

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    print("start to find matches")
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    print("start to computeCorrespondEpilines")
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()


def stero_rectification(data, calib_files_left, calib_files_right, image_size):
    rectify_scale = 0  # 0=full crop, 1=no crop
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(data["cameraMatrix1"], data["distCoeffs1"], data["cameraMatrix2"],
                                                      data["distCoeffs2"], image_size, data["R"], data["T"],
                                                      alpha=rectify_scale) #(240, 320)
    # compute undistortion and rectification; output two maps for remap purpose
    left_maps = cv2.initUndistortRectifyMap(data["cameraMatrix1"], data["distCoeffs1"], R1, P1, image_size, cv2.CV_16SC2)#cv2.CV_16SC2
    right_maps = cv2.initUndistortRectifyMap(data["cameraMatrix2"], data["distCoeffs2"], R2, P2, image_size, m1type=cv2.CV_16SC2)#CV_32FC1
    # print(left_maps[0])
    # print(left_maps[1])
    calib_files = zip(calib_files_left, calib_files_right)
    # calib_files = ('./calibration/CaptureL.JPG', './calibration/CaptureR.JPG')
    # dstmap1_l, dstmap2_l = cv2.convertMaps(map1=left_maps[0], map2=left_maps[1], dstmap1type=cv2.CV_16SC2)
    # dstmap1_r, dstmap2_r = cv2.convertMaps(map1=right_maps[0], map2=right_maps[1], dstmap1type=cv2.CV_16SC2)
    for left_path,right_path in calib_files:
    # left_path = './calibration/CaptureL.JPG'
    # right_path = './calibration/CaptureR.JPG'
    # print(left_path)
    # print(right_path)
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        print(left_img.shape)
        print(right_img.shape)

        # cv2.imshow("rectifed left chess one", left_img)
        # cv2.waitKey(0)
        # cv2.imshow("recified right chess two", left_img[1])
        # left_img = left_img.astype(np.float16)
        # right_img = right_img.astype(np.float16)
        # left_img.convertTo(left_img, cv2.CV_32FC1)
        left_img_remap = cv2.remap(left_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(right_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4) #cv2.INTER_LANCZOS4
        # print("get line")
        # get_epipolar_line(left_img_remap, right_img_remap)
        cv2.imshow("rectifed left chess", left_img_remap)
        cv2.imshow("recified right chess", right_img_remap)
        cv2.waitKey(0)
        # input1 = input("Enter")
    imgL = cv2.cvtColor(left_img_remap, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right_img_remap, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # cv2.imshow('L', imgL)
    # cv2.imshow('R', imgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    disparity = stereo.compute(imgL, imgR)
    print(disparity)
    plt.imshow(disparity, 'gray')
    plt.show()
    cv2.destroyAllWindows()
    return left_img_remap, right_img_remap


    # detector = cv2.FastFeatureDetector_create("HARRIS")
    # extractor = cv2.BOWImgDescriptorExtractor("SIFT")
    # matcher = cv2.DescriptorMatcher_create("BruteForce")

    # for pair in calib_files:
    #     left_kp = detector.detect(left_img_remap)
    #     right_kp = detector.detect(right_img_remap)
    #     l_kp, l_d = extractor.compute(left_img_remap, left_kp)
    #     r_kp, r_d = extractor.compute(right_img_remap, right_kp)
    #     matches = matcher.match(l_d, r_d)
    #     sel_matches = [m for m in matches if abs(l_kp[m.queryIdx].pt[1] - r_kp[m.trainIdx].pt[1]) & lt; 3]


def return_depth(data, calib_files_left, calib_files_right):
    imgL = cv2.imread('./calibration/CaptureL.JPG')
    imgR = cv2.imread('./calibration/CaptureR.JPG')
    # imgL = glob.glob('./calibration/CaptureL.JPG')
    # imgR = glob.glob('./calibration/CaptureR.JPG')
    print(imgR.shape)
    h, w = imgR.shape[:2]
    print(h, w)
    print(imgR.shape)
    imgL=cv2.resize(imgL, (w, h))
    # imgL = imgL[:h, :w, :]
    # print(imgL.shape[,2])
    print(imgL.shape[:2])
    # imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    # imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # cv2.imshow('L', imgL)
    # cv2.imshow('R', imgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgL_rectified, imgR_rectified = stero_rectification(data, calib_files_left, calib_files_right, (h, w))
    imgL_rectified = cv2.cvtColor(imgL_rectified, cv2.COLOR_BGR2GRAY)
    imgR_rectified = cv2.cvtColor(imgR_rectified, cv2.COLOR_BGR2GRAY)
    # calibration = StereoCalibration(input_folder='./stereopi-tutorial-master/stereopi-tutorial-master/calib_result')
    # rectified_pair = calibration.rectify((imgL, imgR))
    # cv2.imshow('Left CALIBRATED', rectified_pair[0])
    # cv2.imshow('Right CALIBRATED', rectified_pair[1])
    # get_epipolar_line(imgL_rectified, imgR_rectified)

    disparity = stereo.compute(imgL_rectified,imgR_rectified)
    # disparity = stereo.compute(rectified_pair[0],rectified_pair[1])
    # print(disparity)
    # # image_depth = disparity*focal_length/distance_between_camera
    print(disparity)
    plt.imshow(disparity,'gray')
    plt.show()
    # cv2.waitKey(0)

'''
# stero pairing

detector = cv2.FeatureDetector_create("HARRIS")
extractor = cv2.DescriptorExtractor_create("SIFT")
matcher = cv2.DescriptorMatcher_create("BruteForce")

for pair in pairs:
    left_kp = detector.detect(pair.left_img_remap)
    right_kp = detector.detect(pair.right_img_remap)
    l_kp, l_d = extractor.compute(left_img_remap, left_kp)
    r_kp, r_d = extractor.compute(right_img_remap, right_kp)
    matches = matcher.match(l_d, r_d)
    sel_matches = [m for m in matches if abs(l_kp[m.queryIdx].pt[1] - r_kp[m.trainIdx].pt[1]) & lt; 3]

'''
'''
# triangulation

for m in sel_matches:
    left_pt = l_kp[m.queryIdx].pt
    right_pt = r_kp[m.trainIdx].pt
    dispartity = abs(left_pt[0] - right_pt[0])
    z = triangulation_constant / dispartity
'''


def main():
    calib_files_left = glob.glob('./stereopi-tutorial-master/stereopi-tutorial-master/pairs/left*.png')
    calib_files_right = glob.glob('./stereopi-tutorial-master/stereopi-tutorial-master/pairs/right*.png')
    data = stero_calibration(calib_files_left, calib_files_right)
    stero_rectification(data, calib_files_left, calib_files_right, (294, 396))
    # return_depth(data,'./calibration/CaptureL.JPG','./calibration/CaptureR.JPG')
    # return_depth(0,0,0)


if __name__ == '__main__':
    main()