import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

import rhombi

HD1080 = (1920, 1080)
HD720 = (1280, 720)
VGA = (640, 480)

(width, height) = HD1080

rhombi.imgWidth = width * 5
rhombi.imgHeight= height * 5
rhombi.sideLength = 25 * 5
rhombi.backgroundColor = (0, 0, 0)
rhombi.rhombusEdgeThickness = 10

hueStart = 0
hueEnd = 180


theta = (np.sqrt(5) - 1) / 2
angleMin = theta
angleMax = angleMin / 2 + 0.5

frameCount = 2400
lineCount = 60

ts = list(np.linspace(0, 1, frameCount))
linesGroup = []
hues = []
saturations = []


for t in ts:
    angle = (1 - t) * angleMin + t * angleMax
    irrationality = rhombi.measureIrrationality(angle)
    hues.append(hueStart * (1 - t) + hueEnd * t)
    saturations.append(int(irrationality * 255))
    radiusMin = np.sin((t + 0.1495) * np.pi * 5) * 0.9999
    print("t = {}, angle = {} irrationality = {}, radiusMin = {}".format(t, angle, irrationality, radiusMin))
    stheta = rhombi.genStar(lineCount, angle = angle, radiusMin = radiusMin, radiusMax = 1, time = t)
    linesGroup.append(stheta)

ims = []
for i, lines, hue, saturation in zip(range(len(linesGroup)), linesGroup, hues, saturations):
    framePct = float(i) / len(linesGroup)
    print("frame {}/{} framePct = {} line count= {}".format(i + 1, len(linesGroup), framePct, len(lines)))
    rhombi.rhombusEdgeColor = (hue, saturation, 255)
    rhombi.rhombusFaceColor1 = (hue, saturation, min(192, saturation))
    rhombi.rhombusFaceColor2 = (hue, saturation, min(128, saturation))
    rhombi.genRhombi(lines)
    img = rhombi.drawImg(lines = lines)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (width, height))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if frameCount > 9:
    imageio.mimwrite(uri = 'rhomboids2.mp4', ims = ims, macro_block_size = None)



