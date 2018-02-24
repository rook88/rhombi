"""
animation.py  --linecountmin 2 --linecountmax 60 --anglemin 0.55 --anglemax 0.9 --framecount 6720 --resolution 720p --testtime 0.56072
"""

import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()


import rhombi

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

lineCountRaw = lineCountMin
lineCount = lineCountMin
lineCountDelta = 1.0 / 24

edgeLength = rhombi.height / lineCountMin * 7

radiusMin = 0.999

ims = []
i = 0

"""
IntroEnd
2:36
156

End
4:40
280 

(/ 156.0 280)
0.5571428571428572

(* 24 280)

6720 frames

0.01 = 67 frames 2.8 seconds

(* 6720 0.00375)
(* 24 0.00375)
(/ 1 0.00375)
(* 24 (/ 1 6720.0))
"""

for t in ts:
    i += 1
    introLeft = np.clip(t * 280 / 157, 0, 1)
    outroLeft = np.clip((t * 280 / 157 - 1) / (280.0 / 157 - 1), 0, 1)
    intro = 1 - introLeft
    outro = 1 - outroLeft
    angle = (1 - t) * angleMin + t * angleMax
    angle2 = angle
    if angle > 0.85:
        angle2 = 1.7 - angle
    if angle > 0.8:
        angle1 = 0.8 + np.sin((angle - 0.8) * 10 * np.pi) / np.pi / 10
        angle = 0.2 * angle2 + 0.8 * angle1
        print angle1, angle2, angle
    lineCount = int(intro * lineCountMin + introLeft * lineCountMax)
    angleDelta = - 8 * intro * intro 
    if outro < 0.2:
        lineCount = 80
    if lineCountRaw < lineCount - lineCountDelta / 2:
        lineCountRaw += lineCountDelta
    if testT or testAngle:
        lineCountRaw = lineCount
    if outro < 0.2:
        normalLength = edgeLength * lineCountMin / lineCountMax * (1 + (0.2 - outro) * 10)
    else:
        normalLength = edgeLength * lineCountMin / lineCountRaw * np.clip(1.9 - intro, 0.9, 1.0) 
        print normalLength, edgeLength, lineCountMin, lineCountRaw
    irrationality = rhombi.measureIrrationality(angle)
    saturation = 255
    radiusMinMax = 0.9999999
    if intro:
        radiusMin = radiusMinMax * (1 - np.clip((introLeft - 0.6) * 5, 0.0, 2.0))
        faceSplit = 0.0
        edgeHue = 180 * introLeft * 4 % 180
        faceHue = (edgeHue + 10 * np.sin(2 * np.pi * 7 * introLeft)) % 180
        edgeColor = (edgeHue, 255, 255)
        faceByDirection = 0
        value = 130
        faceColor = (faceHue, saturation, value - value * (1 - irrationality) * introLeft)
        edgeThickness = 30.0 * intro + 5 * introLeft
        verticeRadius = edgeThickness * (1 + intro * irrationality)
    else:
        edgeThickness = 5
        verticeRadius = edgeThickness
        radiusMin = np.clip(1.02 * np.sin(-np.pi / 2 + 2 * np.pi * outroLeft), -1 * radiusMinMax, radiusMinMax)
        faceSplit = 0.7
        faceByDirection = 1 - irrationality
        value = np.clip(50 * outro + 5000 * outroLeft, 0, 150)
        edgeColor = (0, saturation, np.clip(255 - outroLeft * 20000, 0, 255))
        faceColor = (180 * outroLeft * 4 % 180, saturation, value)

    star = rhombi.genStar(lineCountRaw, angle = angle, radiusMin = radiusMin, radiusMax = 1, angleDelta = angleDelta, normalLength = normalLength)
    r = rhombi.rhombi(star)

    print("t = {}, angle = {} irrationality = {}, radiusMin = {}, angleMin = {},intro = {}, outro = {}, normalLength = {}, edgeLength = {}".format(t, angle, irrationality, radiusMin, angleMin, intro, outro, normalLength, edgeLength))
    print("frame {}/{} framePct = {} line count= {}, saturation = {}, rhombi = {}".format(i, frameCount, float(i) / frameCount, lineCountRaw, saturation, r))
#    for line in star:
#        print line.somePoint / line.direction / 1j

    r.setColors(hue = 180 * intro, value = value, saturation = saturation, faceColor = faceColor, edgeColor = edgeColor, faceSplit = faceSplit, verticeRadius = verticeRadius, edgeThickness = edgeThickness, faceByDirection = faceByDirection)
    img = r.getImg()
#    img = rhombi.drawLines(star)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if frameCount < 10 or testAngle:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if rhombi.outputFile:
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)

cv2.imwrite("last.jpg", img)


