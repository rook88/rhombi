import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

import rhombi

rhombi.imgWidth = 6400
rhombi.imgHeight= 6400
rhombi.sideLength = 240
rhombi.backgroundColor = (255, 93, 15)
rhombi.rhombusEdgeColor = (0, 0, 0)
rhombi.rhombusEdgeThickness = 10

frameCount = 20
lineCount = 40


fs = [(lambda x : np.exp(angle * (x + 0.1 * n))) for n in range(frameCount)]
fs = [rhombi.curry(z) for z in list(np.linspace(0, 1, frameCount + 1))][:-1]


ks = [range(frameCount) for n in range(frameCount)]

ks = [list(np.linspace(1, n + 3, n + 3)) for n in range(frameCount)]
fs = [rhombi.star(13) for n in range(frameCount)]

ts = list(np.linspace(0, 1, frameCount))
ks = [list(np.linspace(1, 21, 21)) for n in range(frameCount)]

fs = [rhombi.star((np.sqrt(5) + 1) / 2 + (1 - t) / 4, 1) for t in ts]
linesGroup = [rhombi.genLines(k, f) for k, f in zip(ks, fs)]

fs = [rhombi.star((np.sqrt(5) + 1) / 2, 1 - t) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]

fs = [rhombi.star((np.sqrt(5) + 1) / 2 + t / 4, 0) for t in ts]
linesGroup += [rhombi.genLines(k, f) for k, f in zip(ks, fs)]


print("------------------------------------------------------------")
print len(linesGroup)
print ks
print [f(1) for f in fs]

ims = []
for i, lines in zip(range(len(linesGroup)), linesGroup):
    print("frame {}/{} line[0] = {}".format(i + 1, len(linesGroup), lines[0]))
    img = rhombi.drawImg(lines = lines)
    img = cv2.resize(img, (640, 640))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if frameCount > 9:
    imageio.mimwrite(uri = 'rhomboids.mp4', ims = ims)#, duration = 0.05)
#imageio.mimwrite(uri = 'rhomboids.gif', ims = ims + list(reversed(ims)), duration = 0.04)

cv2.imwrite('test.jpg', img)
