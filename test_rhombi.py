import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()


import rhombi

frameCount = rhombi.frameCount
lineCount = rhombi.lineCount
edgeLength = rhombi.edgeLength
testT = rhombi.testT
testAngle = rhombi.testAngle
angleMin  = rhombi.angleMin 
angleMax = rhombi.angleMax
ts = rhombi.ts

hueStart = 0
hueEnd = 180

linesGroup = []
hues = []
saturations = []

t08 = (0.8 - angleMin) / (angleMax - angleMin)

for t in ts:
    angle = (1 - t) * angleMin + t * angleMax
    doublePct = np.clip(30 * (angle - 0.75), 0, 1) 
    irrationality = rhombi.measureIrrationality(angle)
    hues.append(hueStart * (1 - t) + hueEnd * t)
    saturations.append(int(irrationality * 128) + 127)
    radiusMin = np.sin((t - t08) * np.pi * 6 - np.pi / 2) * 0.99999
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}".format(t, angle, irrationality, radiusMin))
    star = rhombi.genStar(lineCount, angle = angle, radiusMin = radiusMin, radiusMax = 1, time = t, normalLength = edgeLength, doublePct = doublePct)
    linesGroup.append(star)

ims = []
for i, lines, hue, saturation in zip(range(len(linesGroup)), linesGroup, hues, saturations):
    framePct = float(i) / len(linesGroup)
    r = rhombi.rhombi(lines)
    print("frame {}/{} framePct = {} line count= {}, saturation = {}, rhombi = {}".format(i + 1, len(linesGroup), framePct, len(lines), saturation, r))
    img = r.getImg(hue = hue, saturation = saturation, faceColor = (60, 255, 40), faceSplit = 0.3, edgeColor = (60, 255, 255), verticeRadius = 15)
#    img = r.getImg(edgeColor = (80, 255, 0))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if rhombi.outputFile:
    imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)



