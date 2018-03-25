# slide.py  --linecount 30 --framecount 60 --resolution 720p  --linecountmin 5 --linecountmax 4 --anglemin 900 --anglemax -900  --file slide02.mp4

import numpy as np
import cv2
import imageio
import copy
import rhombi

ts = rhombi.ts
moveOffsetStart = rhombi.angleMin
moveOffsetEnd = rhombi.angleMax
moveDirection = np.sign(moveOffsetEnd - moveOffsetStart)
sideLength = 600
ns = [rhombi.lineCountMin, rhombi.lineCountMax]

stars = []
for n in ns:
    stars.append(rhombi.genStar(rhombi.lineCount, angle = 1.0 / n, angleDelta = 0.0 / 14, normalLength = sideLength, radiusMin = -0.99))

faces = []
i = 0
for star in stars:
    r =  rhombi.rhombiCl(star) 
    r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 1.0, edgeColor = (60,255,0), edgeThickness = 20)
    for faceKey, face in r.faces.items():
        face.order = i
        faces.append(copy.copy(face))
    i += 1


faces.sort(key = lambda f: f.center[0] * moveDirection + f.order)
ims = []

i = 0
frameNumber = 0
for t in ts:
    tsmooth = 0.5 - 0.5 * np.real(np.exp(1j * np.pi * t))
    frameNumber += 1
    offset0 = int((moveOffsetStart + moveOffsetEnd) / 2)
    offset1 = int(moveOffsetStart + tsmooth * (moveOffsetEnd - moveOffsetStart))
    print frameNumber, t, tsmooth, offset0, offset1
    print r
    img = rhombi.getEmptyImg()
    img[:,:] = rhombi.backgroundColor
    for face in faces:
        if face.order == 0:
            offset = offset0 
        else:
            offset = offset1
        face.draw(img, drawAlsoEdges = True, offset = np.array([offset, 0]))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if rhombi.frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)
if rhombi.outputFile:
    imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)
cv2.imwrite("last2.jpg", img)
