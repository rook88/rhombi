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

ts = rhombi.ts

hueStart = 0
hueEnd = 180

linesGroup = []
hues = []
saturations = []
doublePcts = []

t08 = (0.8 - angleMin) / (angleMax - angleMin)

angleDoubles = [2.0 / 3 - 0.5 / 100, 3.0 / 4 - 0.5 / 100, 4.0 / 5 - 0.5 / 100, 5.0 / 6 - 0.5 / 100, 999] 
#angleDoubles = [5.0 / 8 - 0.5 / 100, 3.0 / 4 - 0.5 / 100, 999] 
angleDouble = angleDoubles.pop(0)

radiusMin = 0.8

ts = list(np.linspace(0, 1, frameCount))
ts = [(16.0 - 16 ** (1 - t)) / 15 for t in ts]

print ts


for t in ts:
    angle = (1 - t) * angleMin + t * angleMax
    doublePct = np.clip(100 * (angle - angleDouble), 0, 1) 
    irrationality = rhombi.measureIrrationality(angle)
    hues.append(hueStart * (1 - t) + hueEnd * t)
    saturations.append(int(irrationality * 128) + 127)
    doublePcts.append(doublePct)
#    radiusMin = np.sin((t - t08) * np.pi * 6 - np.pi / 2) * 0.99999
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}, doublePct = {}".format(t, angle, irrationality, radiusMin, doublePct))
    
    star = rhombi.genStar(lineCount, angle = angle, radiusMin = radiusMin, radiusMax = 1, time = t, normalLength = edgeLength, doublePct = doublePct)
    linesGroup.append(star)
    if doublePct == 1.0:
        angleDouble = angleDoubles.pop(0)
	lineCount *= 2
        edgeLength /= 2


ims = []
for i, lines, hue, saturation, doublePct in zip(range(len(linesGroup)), linesGroup, hues, saturations, doublePcts):
    framePct = float(i) / len(linesGroup)
    r = rhombi.rhombi(lines, doublePct)
    print("frame {}/{} framePct = {} line count= {}, saturation = {}, rhombi = {}".format(i + 1, len(linesGroup), framePct, len(lines), saturation, r))
#    for k, e in r.edges.items():
#        print e
    r.setColors(hue = hue, saturation = saturation, faceSplit = 0.0, edgeColor = None, verticeRadius = 0)
    img = r.getImg()
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
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)



