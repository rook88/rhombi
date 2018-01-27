import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

import rhombi

(width, height) = (1920, 1080)

rhombi.imgWidth = width * 5
rhombi.imgHeight= height * 5
rhombi.sideLength = 20 * 5
rhombusEdgeColorStart = np.array([15, 93, 255])
rhombusEdgeColorEnd = np.array([255, 93, 15]) 
rhombi.backgroundColor = (0, 0, 0)
rhombi.rhombusEdgeThickness = 8

linesGroup = []

theta = (np.sqrt(5) - 1) / 2

frameCount = 2400
lineCount = 16

ts = list(np.linspace(0, 1, frameCount))
ks = [list(np.linspace(0, 1, lineCount)) for n in range(frameCount)]
fs = [rhombi.curryDev(angle = 0.2, offset = 0.1 * t) for t in ts]
gs = [rhombi.curryDev(angle = 0.2, offset = 0.1 * t) for t in ts]
print fs
#linesGroup += [rhombi.genLines(k, f, g) for k, f, g in zip(ks, fs, gs)]

#linesGroup.append(rhombi.genStar(25))
for t in ts:
    s5 = rhombi.genStar(5, angle = 1 / 5.0, angleDelta = 2 * np.pi, radiusMin = 0.0009, radiusMax = 0.0010, center = np.exp(10 * rhombi.a2pi * t) * 0.8, time = t)
    s7 = rhombi.genStar(7, angle = 1 / 7.0, angleDelta = 2 * np.pi, radiusMin = 0.0009, radiusMax = 0.0010, center = np.exp(10 * theta * rhombi.a2pi * t) * 0.9, time = t)
    stheta = rhombi.genStar(61, angleDelta = - np.pi, radiusMin = -1 + 2 * t, radiusMax = 1, time = t)
    linesGroup.append(s5 + s7 + stheta)
#linesGroup.append(rhombi.genStar(25, radiusMin = -1.0))
#linesGroup.append(rhombi.genStar(25, angle = (np.sqrt(5) - 1) / 2, radiusMin = -1.0))
#linesGroup.append(rhombi.genStar(25, angle = (np.sqrt(5) - 1) / 2, radiusMin = 1.0))

#print linesGroup
#print linesGroup[0]

ims = []
for i, lines in zip(range(len(linesGroup)), linesGroup):
    framePct = float(i) / len(linesGroup)
    print("frame {}/{} framePct = {} line[0] = {}".format(i + 1, len(linesGroup), framePct, lines[0]))
    rhombi.rhombusEdgeColor = rhombusEdgeColorStart * (1 - framePct) + rhombusEdgeColorEnd * framePct
    img = rhombi.drawImg(lines = lines)
#    img = rhombi.drawLines(lines = lines)
    img = cv2.resize(img, (width, height))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if frameCount > 9:
    imageio.mimwrite(uri = 'rhomboids2.mp4', ims = ims, macro_block_size = None)#, duration = 0.05)
#imageio.mimwrite(uri = 'rhomboids.gif', ims = ims + list(reversed(ims)), duration = 0.04)

cv2.imwrite('test.jpg', img)
