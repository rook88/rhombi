import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

import rhombi

(width, height) = (1920, 1080)

rhombi.imgWidth = width * 5
rhombi.imgHeight= height * 5
rhombi.sideLength = 22 * 5
rhombusEdgeColorStart = np.array([0, 255, 255])
rhombusEdgeColorEnd = np.array([180, 255, 255]) 
rhombi.backgroundColor = (0, 0, 0)
rhombi.rhombusEdgeThickness = 10

linesGroup = []

theta = (np.sqrt(5) - 1) / 2
angleMax = 1 - theta
angleMin = angleMax / 2

frameCount = 2400
lineCount = 61

ts = list(np.linspace(0, 1, frameCount))

colors = []
for t in ts:
    angle = (1 - t) * angleMin + t * angleMax
    irrationality = rhombi.measureIrrationality(angle)
    color = rhombusEdgeColorStart * (1 - t) + rhombusEdgeColorEnd * t
    color[1] = int(irrationality * 255)
    colors.append(color)
    radiusMin = np.sin(theta / frameCount + t * np.pi * 5) * 0.99
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}".format(t, angle, irrationality, radiusMin))
    stheta = rhombi.genStar(lineCount, angle = angle, radiusMin = radiusMin, radiusMax = 1, time = t)
    linesGroup.append(stheta)

#linesGroup.append(rhombi.genStar(25, radiusMin = -1.0))
#linesGroup.append(rhombi.genStar(25, angle = (np.sqrt(5) - 1) / 2, radiusMin = -1.0))
#linesGroup.append(rhombi.genStar(25, angle = (np.sqrt(5) - 1) / 2, radiusMin = 1.0))

#print linesGroup
#print linesGroup[0]

ims = []
for i, lines, color in zip(range(len(linesGroup)), linesGroup, colors):
    framePct = float(i) / len(linesGroup)
    print("frame {}/{} framePct = {} line count= {}".format(i + 1, len(linesGroup), framePct, len(lines)))
    rhombi.rhombusEdgeColor = color
    img = rhombi.drawImg(lines = lines)
#    img = rhombi.drawLines(lines = lines)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
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
