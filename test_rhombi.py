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

edgeLength = rhombi.height / lineCountMin * 5

radiusMin = 0.80

ims = []
i = 0
for t in ts:
    i += 1
    angle = (1 - t) * angleMin + t * angleMax
    if lineCountRaw < lineCount - lineCountDelta / 2:
        lineCountRaw += lineCountDelta
    lineCount = int((1 - t) * lineCountMin + t * (lineCountMax + 1))
    irrationality = rhombi.measureIrrationality(angle)
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}".format(t, angle, irrationality, radiusMin))
    star = rhombi.genStar(lineCountRaw, angle = angle, radiusMin = radiusMin, radiusMax = 1, time = t, normalLength = edgeLength * lineCountMin / lineCountRaw)
    r = rhombi.rhombi(star)
    print("frame {}/{} framePct = {} line count= {}, saturation = {}, rhombi = {}".format(i, frameCount, float(i) / frameCount, lineCountRaw, "?", r))
    split = np.clip(t - 0.5, 0.0, 0.3)
    r.setColors(hue = 180 * t, value = 50, faceSplit = split, verticeRadius = 0)
    img = r.getImg()
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if rhombi.outputFile:
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)



