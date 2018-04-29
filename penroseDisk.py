#C:\Users\Dell\Documents>python git\rhombi\rhombi\penroseDisk.py --edgelength 50 --resolution 800x800  --linecount 20  --framecount 60 --file penroseDisk.mp4

import numpy as np
import cv2
import imageio
import rhombi

frameCount = rhombi.frameCount
lineCount = rhombi.lineCount
edgeLength = rhombi.edgeLength * 5.0
lineCountMin = rhombi.lineCountMin
lineCountMax = rhombi.lineCountMax

theta = np.sqrt(5) * 0.5 - 0.5
angle = 0.2

#lines = rhombi.genStar(rhombi.lineCount, angle = angle, radiusMin = -0.99, normalLength = edgeLength)

lineDistanceA = np.sqrt(theta * 3 / 2)
lineDistanceB = 1 / lineDistanceA

lineDistances = [
    1
    ,lineDistanceB
    ,lineDistanceA
    ,lineDistanceA
    ,lineDistanceB]

lines = []
for i in range(5):
    for j in range(lineCount):
        direction = np.exp(2j * np.pi * angle * i)
        angleLine = angle * i
        offset = lineDistances[i] * (j - lineCount / 2) + 0.001
        l = rhombi.line(angle = angleLine, offset = offset, normalLength = edgeLength)
        lines.append(l)

print theta
print lineDistanceA
print lineDistanceB
print lineDistanceB * 3 / np.cos(2 * np.pi / 20)
print lineDistanceA * 2 / np.cos(2 * np.pi / 20 * 3)

r = rhombi.rhombiCl(lines)

#for faceKey, face in r.faces.items():
#    print face

print r

r.setColors(hue = 60, value = 255, saturation = 255, faceByDirection = 1.0, faceColor = (30, 255, 255), edgeColor = (60,255,0), edgeThickness = 10)

#r.setColors(hue = 180 * intro, value = value, saturation = saturation, faceColor = faceColor, edgeColor = edgeColor, faceSplit = faceSplit, verticeRadius = verticeRadius, edgeThickness = edgeThickness, faceByDirection = faceByDirection)

offsetMax = lineDistanceB * 3 / np.cos(2 * np.pi / 20)
a = np.cos(2 * np.pi / 20)
b = np.cos(2 * np.pi / 20 * 3)

offsetMax = - (4 * b + 6 * a) * edgeLength

print offsetMax
#exit(0)

rhombi.diskRadius = 1200
rhombi.diskOffset = offsetMax 

ims = []
for fr in range(frameCount):
    print "frame Number {}".format(fr)
    rhombi.diskOffset = (fr * offsetMax) / frameCount 
    img = r.getImg()
#    img = rhombi.drawLines(lines)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, (rhombi.width, rhombi.height))
    if rhombi.frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        ims.append(img)

if rhombi.outputFile:
    if 'jpg' in rhombi.outputFile:
        cv2.imwrite(rhombi.outputFile, img)
    else:
        imageio.mimwrite(uri = rhombi.outputFile, ims = ims, macro_block_size = None, fps = 24)

cv2.imwrite("lastPenrose.jpg", img)
