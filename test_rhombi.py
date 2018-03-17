import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()


import rhombi

"""
frameCount = rhombi.frameCount
lineCount = rhombi.lineCount
edgeLength = rhombi.edgeLength * 5.0
testT = rhombi.testT
testAngle = rhombi.testAngle
angleMin  = rhombi.angleMin 
angleMax = rhombi.angleMax
lineCountMin = rhombi.lineCountMin
lineCountMax = rhombi.lineCountMax
ts = rhombi.ts
print ts
"""

theta = np.sqrt(5) / 2 - 1
ims = []
i = 0

ts = rhombi.ts


directions3 = [np.exp(np.pi * 2j * (i / 3.0 + 1 / 13.0)) for i in range(3)]
directions5 = [np.exp(np.pi * 2j * (i / 5.0 + 0.03)) for i in range(5)]

p0 = -3j
p1 = +3j

lines3 = [rhombi.line(somePoint = p0, directionPoint = d, normalLength = 500) for d in directions3]
lines5 = [rhombi.line(somePoint = p1, directionPoint = d, normalLength = 500) for d in directions5]


points = []
for l3 in lines3:
#    points5 = []
    for l5 in lines5:
        p = rhombi.intersection(l3, l5)
        if np.real((p - p0) / l3.direction) > 0:
            sign3 = 1
        else:
            sign3 = -1
        if np.real((p - p1) / l5.direction) > 0:
            sign5 = 1
        else:
            sign5 = -1
        print p, sign3, sign5
        points.append((p, sign3, sign5))
#    points.append(points5)

frameNumber = 0
for t in ts:
#    pt = p0 * (1 - np.sin(np.pi * t)) + 0.999 * np.sin(np. pi * t) * p1
    angle = np.exp(2j * np.pi * t)
    anglen = np.real(angle) #+ np.imag(angle) / 5
    pt = p0 * anglen
    offsetT = abs(pt - p0) / abs(p0 - p1)
    frameNumber += 1
    print frameNumber, pt
    lines = []
    offset = 0
    i = 0
    for ptuple in points:
        p, s3, s5 = ptuple
        offset = offsetT * (i / 5 - 1) + (1 - offsetT) * (i % 5 - 2) + 0.33
        sign = offsetT * s5 + (1 - offsetT) * s3
        i += 1
        print offsetT, offset
        direction = (p - pt) / abs(p - pt)
#        if np.real(direction) <= 0:
#            direction = -direction
        offset = direction / 1j * offset * 1.0 * sign
        line = rhombi.line(somePoint = pt + offset, directionPoint = direction, normalLength = 200)
        lines.append(line)
    if rhombi.parDrawLines:
        img = rhombi.drawLines(lines)
    else:
        r = rhombi.rhombi(lines)
        r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 2.0, edgeColor = (0,0,0))
        img = r.getImg(smooth = False)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
#    img = img[:, 280:1000, :]
    if rhombi.frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if rhombi.outputFile:
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)

cv2.imwrite("last2.jpg", img)


