import cv2
import numpy as np



def main():
    im1 = cv2.imread('plots/sa_t1_03_250000_pic.png')
    im2 = cv2.imread('plots/sa_t1_06_250000_pic.png')
    im3 = cv2.imread('plots/sa_t1_09_250000_pic.png')

    im_h = cv2.hconcat([im1, im2, im3])

    cv2.imwrite('plots/sa_t1_concat.png', im_h)

    im1 = cv2.imread('plots/sa_t2_03_2000_pic.png')
    im2 = cv2.imread('plots/sa_t2_06_2000_pic.png')
    im3 = cv2.imread('plots/sa_t2_09_2000_pic.png')

    im_h = cv2.hconcat([im1, im2, im3])

    cv2.imwrite('plots/sa_t2_concat.png', im_h)

    im1 = cv2.imread('plots/sa_t3_20_10_pic.png')
    im2 = cv2.imread('plots/sa_t3_200_100_pic.png')
    im3 = cv2.imread('plots/sa_t3_2000_1000_pic.png')

    im_h = cv2.hconcat([im1, im2, im3])

    cv2.imwrite('plots/sa_t3_concat.png', im_h)

if __name__ == '__main__':
    main()