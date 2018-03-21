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

"""
directions3 = [np.exp(np.pi * 2j * (i / 3.0 + 1 / 13.0)) for i in range(15)]
directions5 = [np.exp(np.pi * 2j * (i / 5.0 + 0.03)) for i in range(15)]

p1 = +1j

lines3 = [rhombi.line(somePoint = d * 1j, directionPoint = d, normalLength = 500) for d in directions3]
lines5 = [rhombi.line(somePoint = d * 1j, directionPoint = d, normalLength = 500) for d in directions5]
"""
p0 = (+1j + 1) / 10000.0
p1 = (-3j + 2) / 10000.0
p2 = (+2j - 3) / 10000.0


stars = []
stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / 3, center = p0, radiusMin = -0.99))
stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / 7, center = p2, radiusMin = -0.99))
stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / 4, center = p1, radiusMin = -0.99))
stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / 5, radiusMin = -0.99))

starCount = len(stars)
print "Star count = {}".format(starCount)
angleDelta = np.exp(2j * np.pi / starCount)
print "Angle delta = {}".format(angleDelta)

sideLength = 500

frameNumber = 0
for t in ts:
    angle = np.exp(2j * np.pi * t)
    frameNumber += 1
    print frameNumber, t, angle
    lines = []
    i = 0
    for star in stars:
        a = angle * angleDelta ** i
        visible = np.clip(np.real(a), 0, 1) ** 2
#        print "i = {}, a = {}, visible = {}".format(i, a, visible)
        for line in star:
            line.normalLength = 0.01 + sideLength * visible
            lines.append(line)
        i += 1

    if rhombi.parDrawLines:
        img = rhombi.drawLines(lines)
    else:
        r = rhombi.rhombi(lines)
        r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 1.0, edgeColor = (0,0,0), edgeThickness = 25, faceOnlyRhombus = True)
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


