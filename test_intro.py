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

linesGroup = []
saturations = []

t08 = (0.8 - angleMin) / (angleMax - angleMin)

lineCountFrames = list(np.linspace(2, lineCount, frameCount))
for lineCountFrame in lineCountFrames:
    angle = testAngle
    irrationality = rhombi.measureIrrationality(angle)
    saturations.append(int(irrationality * 255))
    intro = []
    r = 1.0
    for lineNumber in range(int(np.ceil(lineCountFrame))):
        pct = np.clip(lineCountFrame - lineNumber, 0, 1)
        print "lines in frame = {}, line number = {}, pct = {}".format(lineCountFrame, lineNumber, pct)
        n = lineNumber
        a = angle * n * rhombi.a2pi
        r += 1
        sp = np.exp(a) * r 
        d = 1j * np.exp(a)
        line1 = rhombi.line(sp, d, rhombi.edgeLength, visible = pct)
        intro.append(line1)
    linesGroup.append(intro)

#    print("t = {}, angle = {} irrationality = {}, radiusMin = {}".format(t, angle, irrationality, radiusMin))

hue = 30
ims = []
for i, lines, saturation in zip(range(len(linesGroup)), linesGroup, saturations):
    framePct = float(i) / len(linesGroup)
    print("frame {}/{} framePct = {} line count= {}".format(i + 1, len(linesGroup), framePct, len(lines)))
    rhombi.rhombusEdgeColor = (hue, saturation, 255)
    rhombi.rhombusFaceColor1 = (hue, 255, 255)
    rhombi.rhombusFaceColor2 = (hue, 255, 255)
    rhombi.genRhombi(lines)
    img = rhombi.drawImg(lines = lines)
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



