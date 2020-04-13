import numpy as np
import cv2
from matplotlib import pyplot as plt

def return_depth(focal_length, distance_between_camera):
    imgL = cv2.imread('./calibration/CaptureL.JPG')
    imgR = cv2.imread('./calibration/CaptureR.JPG')
    h, w = imgR.shape[:2]
    print(h, w)
    print(imgR.shape)
    imgL=cv2.resize(imgL, (w, h))
    # imgL = imgL[:h, :w, :]
    # print(imgL.shape[,2])
    print(imgL.shape[:2])
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # cv2.imshow('L', imgL)
    # cv2.imshow('R', imgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    disparity = stereo.compute(imgL,imgR)
    # print(disparity)
    # image_depth = disparity*focal_length/distance_between_camera
    print(disparity)
    plt.imshow(disparity,'gray')
    plt.show()


def load_data(load_file):
    # Load previously saved data
    with np.load(load_file) as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    return mtx, dist


def calculate_focal_length(f_pixel_x, f_pixel_y):
    f_pixel = (f_pixel_x + f_pixel_y) / 2


def main():
    load_file = './calibration_output.npz'
    mtx, dist = load_data(load_file)
    # f_pixel_x = mtx[0][0]
    # f_pixel_y = mtx[1][1]
    # calculate_focal_length(f_pixel_x, f_pixel_y)
    focal_length = 4.96 # mm
    distance_between_camera = 32 #mm
    return_depth(focal_length, distance_between_camera)
    # 22.2 - distance_between_camera_and_book, 10mm - distance between camera


if __name__ == '__main__':
    main()