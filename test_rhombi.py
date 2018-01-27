import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

import rhombi

rhombi.imgWidth = 1280 * 5
rhombi.imgHeight= 960 * 5
rhombi.sideLength = 20 * 5
rhombusEdgeColorStart = np.array([15, 93, 255])
rhombusEdgeColorEnd = np.array([255, 93, 15]) 
rhombi.backgroundColor = (0, 0, 0)
rhombi.rhombusEdgeThickness = 8

lineCount = 70

linesGroup = []

theta = (np.sqrt(5) - 1) / 2

frameCount = 60
ts = list(np.linspace(0, 1, frameCount))
ks = [list(np.linspace(-1, 1, lineCount)) for n in range(frameCount)]

fs = [rhombi.curryNew(angle = theta, offset = 1, offsetAngle = 0.2 + t) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]

fs = [rhombi.curryNew(angle = theta, offset = 1 - t, offsetAngle = 1.2 + t) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]

fs = [rhombi.curryNew(angle = theta, offset = 0, offsetAngle = 2.2 + t) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]

"""
Toimii:
frameCount = 4
ts = list(np.linspace(0, 1, frameCount))
ks = [list(np.linspace(-1, 1, lineCount)) for n in range(frameCount)]
fs = [rhombi.curryNew(angle = theta, offsetAngle = t) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]
"""

print("------------------------------------------------------------")
print len(linesGroup)
print ks
print [f(1) for f in fs]

ims = []
for i, lines in zip(range(len(linesGroup)), linesGroup):
    framePct = float(i) / len(linesGroup)
    print("frame {}/{} framePct = {} line[0] = {}".format(i + 1, len(linesGroup), framePct, lines[0]))
    rhombi.rhombusEdgeColor = rhombusEdgeColorStart * (1 - framePct) + rhombusEdgeColorEnd * framePct
    img = rhombi.drawImg(lines = lines)
#    img = rhombi.drawLines(lines = lines)
    img = cv2.resize(img, (1280, 960))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if frameCount > 9:
    imageio.mimwrite(uri = 'rhomboids.mp4', ims = ims)#, duration = 0.05)
#imageio.mimwrite(uri = 'rhomboids.gif', ims = ims + list(reversed(ims)), duration = 0.04)

cv2.imwrite('test.jpg', img)
