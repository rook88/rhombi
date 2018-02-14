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

hueStart = 0
hueEnd = 180

linesGroup = []
hues = []
saturations = []

t08 = (0.8 - angleMin) / (angleMax - angleMin)



#ts = list(np.linspace(0, 1, frameCount))
#ts = [(16.0 - 16 ** (1 - t)) / 15 for t in ts]

print ts

lineCountRaw = lineCountMin
lineCount = lineCountMin
lineCountDelta = 1.0 / 24

edgeLength = rhombi.height / lineCountMin * 7

radiusMin = 0.99

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

"""

angleDelta = 0.0
angleDelta2 = 0.01

for t in ts:
    i += 1
    introLeft = np.clip(t * 280 / 156, 0, 1)
    outroLeft = np.clip((t * 280 / 156 - 1) / (280.0 / 156 - 1), 0, 1)
    intro = 1 - introLeft
    outro = 1 - outroLeft
    edgeThickness = 30.0 * intro + 5 * introLeft  
    verticeRadius = edgeThickness * (1 + intro)
    faceSplit = np.clip(introLeft * introLeft * intro * 8, 0.0, 0.7)
    angle = (1 - t) * angleMin + t * angleMax
    lineCount = int(intro * lineCountMin + introLeft * lineCountMax)
    angleDelta2 *= 0.999 
    angleDelta += angleDelta2
    if outro < 0.1:
        lineCount = 80
    if lineCountRaw < lineCount - lineCountDelta / 2:
        lineCountRaw += lineCountDelta
    if testT:
        lineCountRaw = lineCount
    if outro < 0.1:
        normalLength = edgeLength * lineCountMin / lineCountMax * (1 + (0.1 - outro) * 10)
    else:
        normalLength = edgeLength * lineCountMin / lineCountRaw 
        print normalLength, edgeLength, lineCountMin, lineCountRaw
    irrationality = rhombi.measureIrrationality(angle)
    radiusMin = 0.99 * (1 - np.clip(3 * t / 2 - 0.5, 0.0, 1.0))
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}, intro = {}, outro = {}, normalLength = {}, edgeLength = {}".format(t, angle, irrationality, radiusMin, intro, outro, normalLength, edgeLength))
    star = rhombi.genStar(lineCountRaw, angle = angle, radiusMin = radiusMin, radiusMax = 1, angleDelta = angleDelta, normalLength = normalLength)
    r = rhombi.rhombi(star)
    if intro:
        edgeColor = (180 * intro, 255, 255)
        faceByDirection = 0
        value = 50
        saturation = 255
        faceColor = (180 * intro, saturation, value)
    else:
        faceByDirection = 1 - irrationality
        value = np.clip(50 * outro + 10000 * outroLeft, 0, 150)
        saturation = np.clip(255 - 10000 * outroLeft, 0, 255)
        edgeColor = (0, saturation, np.clip(255 - outroLeft * 10000, 0, 255))
        faceColor = (0, saturation, value)

    print("frame {}/{} framePct = {} line count= {}, saturation = {}, rhombi = {}".format(i, frameCount, float(i) / frameCount, lineCountRaw, saturation, r))

    r.setColors(hue = 180 * intro, value = value, saturation = saturation, faceColor = faceColor, edgeColor = edgeColor, faceSplit = faceSplit, verticeRadius = verticeRadius, edgeThickness = edgeThickness, faceByDirection = faceByDirection)
    img = r.getImg()
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



