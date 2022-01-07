import cv2.cv2 as cv
import numpy as np
from glob import glob


# register
#   detect and discr each keypoint
#   match each corresponding
#   sortfor Bist amtes
# Reprojection
#   find Homography
#   reproject each
# blinding
#   use distance transfor
def isoscaling(img, isoscale=500):
    md = min(img.shape[:2])
    w, h = [int(isoscale * (max(md, x) / md)) for x in img.shape[:2][::-1]]
    return cv.resize(img, (w, h))


def imshow(img, name='dummy', isoscale=500):
    img = isoscaling(img, isoscale)
    cv.namedWindow(name)
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyWindow(name)


def getmatched_kps(kps1, disc1,
                   kps2, disc2,
                   k=2, acc_dist=True, num=200):
    matchs = cv.BFMatcher().knnMatch(disc1, disc2, k)
    goodm, dist_acc = [], 0
    for m, n in matchs:
        if m.distance < .75 * n.distance:
            dist_acc += m.distance
            goodm.append(m)
    goodm = sorted(goodm, key=lambda x: x.distance)
    mkps1 = np.array([kps1[m.queryIdx].pt for m in goodm[:num]]).reshape(-1, 1, 2)
    mkps2 = np.array([kps2[m.trainIdx].pt for m in goodm[:num]]).reshape(-1, 1, 2)
    return (mkps1, mkps2, dist_acc) if acc_dist else (mkps1, mkps2,)


def getHomo_andprojplane(org_kp, org_shape, Rlt_kp, Rlt_shape):
    Hr, s = cv.findHomography(Rlt_kp, org_kp, cv.RANSAC, 5.0)
    rup = lambda x: x if int(x) == x else int(x) + 1
    oh, ow = org_shape[:2]
    rh, rw = Rlt_shape[:2]
    cor = np.array([[0, 0, 1], [0, rh, 1],
                    [rw, rh, 1], [rw, 0, 1]]).T
    tc = np.matmul(Hr, cor, )
    tc /= tc[2]
    dw, dh = [-min(tc[i].min(), 0) for i in [0, 1]]
    mw, mh = [max(tc[i].max(), x) for i, x in enumerate([ow, oh])]
    Ho = np.array([[1, 0, dw], [0, 1, dh], [0, 0, 1]])
    plane_dims = (rup(dw + mw), rup(dh + mh))
    return Ho, np.matmul(Ho, Hr), plane_dims


def get_distMask(img, H, plane_dim):
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    mask = cv.warpPerspective(mask, H, plane_dim)
    mask = cv.distanceTransform(mask, cv.DIST_L2, 3)
    return mask[..., np.newaxis]


def getMasks(img1, Ho, img2, Hr, plane_dims):
    msk1 = get_distMask(img1, Ho, plane_dims) + 1
    msk2 = get_distMask(img2, Hr, plane_dims) + 1
    tmsk = msk1 + msk2
    msk1 /= tmsk
    msk2 /= tmsk
    return msk1, msk2


def stitch2Images(img1, img2):
    kps1, disc1 = cv.xfeatures2d.SIFT_create().detectAndCompute(img1, None)
    kps2, disc2 = cv.xfeatures2d.SIFT_create().detectAndCompute(img2, None)
    mk1, mk2 = getmatched_kps(kps1, disc1, kps2, disc2, acc_dist=False)
    Ho, Hr, plane_dims = getHomo_andprojplane(mk1, img1.shape,
                                              mk2, img2.shape)
    msk1, msk2 = getMasks(img1, Ho, img2, Hr, plane_dims)
    out = cv.warpPerspective(img1, Ho, plane_dims, ) * msk1
    out += cv.warpPerspective(img2, Hr, plane_dims, ) * msk2
    return out


if __name__ == '__main__':
    pths = glob('input_image/grail/*.jpg')[:10]
    img = cv.imread(pths[0], )
    for p in pths[1:]:
        temp = cv.imread(p, )
        img = stitch2Images(img, temp)
        img /= img.max()
        img = np.array(img * 255, dtype=np.uint8)
    imshow(img / img.max())
    cv.imwrite('out.jpg', (img / img.max()) * 255)
