import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()


import rhombi

"""
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
"""

ims = []
i = 0

ts = rhombi.ts

lines = []


#lines.append(rhombi.line(offset = -0.3, angle = 0.81, normalLength = 400, visible = 0.5))

theta = np.sqrt(5) / 2 - 1
frameNumber = 0
for t in ts:
    frameNumber += 1
    t1 = np.sin(np.pi * t) ** 2
    angle = 0.40  * (1 - t1) + 0.333333333 * t1
    print frameNumber
    lines = []
    for i in range(rhombi.lineCount):
        line = rhombi.line(offset = i - (1 - t1) * (i % 5) - 1.8, angle = i * angle, normalLength = 500)
        lines.append(line)
    r = rhombi.rhombi(lines)
    r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 1.0)
    img = r.getImg(smooth = True)
#    img = rhombi.drawLines(lines)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    img = img[:, 280:1000, :]
    if rhombi.frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if rhombi.outputFile:
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)

cv2.imwrite("last2.jpg", img)


