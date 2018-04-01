# pieces.py  --linecount 42 --framecount 90 --resolution 960x960 --linecountmin 1 --linecountmax 2  --file pieces.mp4 

import numpy as np
import cv2
import imageio
import copy
import rhombi
import hilbert 

ts = rhombi.ts
moveOffsetStart = rhombi.angleMin
moveOffsetEnd = rhombi.angleMax
moveDirection = np.sign(moveOffsetEnd - moveOffsetStart)
sideLengths = [0, 0, 0, 610, 600, 660, 0, 670]
sideLengths = [0, 0, 0, 325, 300, 330, 0, 335]
#ns = [rhombi.lineCountMin, rhombi.lineCountMax]
ns = [4, 5, 3, 7]

stars = []
for n in ns:
    stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / n, angleDelta = 0.0 / 24, normalLength = sideLengths[int(n)], radiusMin = -0.99))

faceGroups = []
i = 0
for n, star in zip(ns, stars):
    r =  rhombi.rhombiCl(star) 
    r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 1.0, edgeColor = (60,255,0), edgeThickness = 20)
    faces = []
    for faceKey, face in r.faces.items():
        x, y = face.center
        if x >= 0 and x < rhombi.imgWidth and y >= 0 and y < rhombi.imgHeight:
            f = face.getCopy()
            f.sideLength = sideLengths[n]
            f.orderFilter = y
            f.orderHilbert = hilbert.h2i((x + 0.5) /rhombi.imgWidth, (y + 0.5) / rhombi.imgHeight)
            faces.append(f)

    i += 1
    faces.sort(key = lambda(x): x.orderFilter)
    faces = faces[:256]
    faces.sort(key = lambda(x): x.orderHilbert)
    faceGroups.append(faces)
    print "faceGroup {} has {} faces".format(i, len(faces))


ims = []


def drawFace(img, f):
    dir = np.exp(1j * f.direction) * f.sideLength
    an = np.exp(1j * f.angle / 2)
    length1 = np.real(an)
    length2 = np.imag(an)
    dx1 = int(np.real(dir * length1))
    dy1 = -int(np.imag(dir * length1))
    dx2 = int(np.real(1j * dir * length2))
    dy2 = -int(np.imag(1j * dir * length2))
    x, y = f.center
    v1 = (x + dx1, y + dy1)
    v2 = (x + dx2, y + dy2)
    v3 = (x - dx1, y - dy1)
    v4 = (x - dx2, y - dy2)
    h = f.hue
    cv2.fillConvexPoly(img, np.array([v1, v2, v3, v4]), (h, 255, 255))
    cv2.line(img = img, pt1 = v1, pt2 = v2, color = (0,0,0), thickness = 20)
    cv2.line(img = img, pt1 = v2, pt2 = v3, color = (0,0,0), thickness = 20)
    cv2.line(img = img, pt1 = v3, pt2 = v4, color = (0,0,0), thickness = 20)
    cv2.line(img = img, pt1 = v4, pt2 = v1, color = (0,0,0), thickness = 20)

def avgFace(f1, f2, t):
    ret = type('', (), {})() 
    ret.center = (t * f1.center + (1 - t) * f2.center).astype(int)
    ret.direction = t * f1.direction + (1 - t) * f2.direction 
    ret.angle = t * f1.angle + (1 - t) * f2.angle
    ret.hue = t * f1.color[0] + (1 - t) * f2.color[0] 
    ret.sideLength = t * f1.sideLength + (1 - t) * f2.sideLength 
    return ret

frameNumber = 0
for t in ts:
    tsmooth = 0.5 - 0.5 * np.real(np.exp(1j * np.pi * t))
    frameNumber += 1
    print frameNumber, t, tsmooth
    print r
    img = rhombi.getEmptyImg()
    img[:,:] = rhombi.backgroundColor

    faces1 = faceGroups[int(rhombi.lineCountMin) - 1]
    faces2 = faceGroups[int(rhombi.lineCountMax) - 1]
    n = ns[int(rhombi.lineCountMin) - 1]
    i = 0
    for f1, f2 in zip(faces1, faces2):
#        print "1, i = {}, center = {}, direction = {}, angle = {}".format(n, f1.center, f1.direction, f1.angle)
#        print "2, i = {}, center = {}, direction = {}, angle = {}".format(n, f2.center, f2.direction, f2.angle)
        f = avgFace(f1, f2, tsmooth)
#        print "3, i = {}, center = {}, direction = {}, angle = {}".format(n, f.center, f.direction, f.angle)
        drawFace(img, f)
#        if i > 1:
#            cv2.line(img = img, pt1 = (x, y), pt2 = (xPrev, yPrev), color = (0,0,255), thickness = 20)
        xPrev, yPrev = x, y
        i += 1
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if rhombi.frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)
if rhombi.outputFile:
    imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)
cv2.imwrite("last3.jpg", img)
