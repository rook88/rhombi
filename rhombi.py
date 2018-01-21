import numpy as np
import cv2
import imageio
#imageio.plugins.ffmpeg.download()

fullAngle = 2 * np.pi
rightAngle = 0.5 * np.pi
origin = 0 + 0j

class line():
    def __init__(self, somePoint = None, directionPoint = None):
        if somePoint <> None and directionPoint <> None:
            self.somePoint = somePoint
            self.direction = directionPoint / abs(directionPoint)
            self.angle = np.imag(np.log(directionPoint))
    def getNormal(self):
        return self.direction / np.exp(rightAngle * 1j)
    def __str__(self):
        return "Line somePoint = {} angle = {} direction = {}".format(self.somePoint, self.angle / fullAngle, self.direction)

#print fullAngle

lines = []
lines.append(line(0 + 0j, 0 + 1j))
lines.append(line(1 + 0j, 0 + 1j))
lines.append(line(0 + 1j, 1 + 0.5j))
lines.append(line(0 + 1.2j, 1 + 0.5j))
lines.append(line(0 + 0j, 1 + 0j))
lines.append(line(0.5 + 0j, 1 + 4j))

#lines.append(line(1 + 1j, -2 + 1j))
#lines.append(line(2 + 1j, -2 - 1j))

lines = []
number = 9
for i in range(number):
    angle = fullAngle * 1.0j * i / number
    lines.append(line(np.exp(angle), np.exp(rightAngle * 1j + angle)))
    lines.append(line(np.exp(angle) * 0.1, np.exp(rightAngle * 1j + angle)))
    lines.append(line(np.exp(angle) * -2, np.exp(rightAngle * 1j + angle)))


lines = []
number = 25
theta1 = (np.sqrt(5) - 1) / 2 * fullAngle
theta2 = 0.25 * fullAngle
angle = 0
for i in range(number):
    angle += theta1 * 1j
    lines.append(line(np.exp(angle) * (1 + 0.00001 * i), np.exp(rightAngle * 1j + angle)))
for i in range(number * 2):
    angle += theta2 * 1j
    lines.append(line(np.exp(angle) * (1 + 0.00001 * i), np.exp(rightAngle * 1j + angle)))

def z2xy(z):
    return (np.real(z), np.imag(z))

def intersection(l1, l2):
    (x1, y1) = z2xy(l1.somePoint)
    (x2, y2) = z2xy(l1.somePoint + l1.direction)
    (x3, y3) = z2xy(l2.somePoint)
    (x4, y4) = z2xy(l2.somePoint + l2.direction)
    if ((x1- x2)*(y3- y4) - (y1 - y2)*(x3 - x4)) == 0:
        return None
    else:
        ret = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / ((x1- x2)*(y3- y4) - (y1 - y2)*(x3 - x4)) + ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / ((x1- x2)*(y3- y4) - (y1 - y2)*(x3 - x4)) * 1j
        if np.abs(ret) > 10000.0:
            return None
        else:
            return ret

def position(l, x):
    if np.imag((x - l.somePoint) / l.direction) >= 0:
        return 0
    else:
        return 1

"""
for l in lines:
    print(l)
    print(position(l, origin))
    print(position(l, 3))
"""
    
def genRhomboidFaceKeys(lines):
    ret = {}
    previousIntersections = []
    for i1 in range(len(lines)):
        for i2 in range(i1 + 1, len(lines)):
            p = intersection(lines[i1], lines[i2])
            if p <> None:
                if p in previousIntersections:
                    raise ValueError("multiple lines intersect at same point", p)
                previousIntersections.append(p)
                retKey = ""
                for i3 in range(len(lines)):
                    if i3 == i1:
                        retKey += "a"
                    elif i3 == i2:
                        retKey += "b"
                    else:
                        retKey += str(position(lines[i3], p))
                ret[retKey] = (i1, i2, p)
    return ret

def getRhomboidVertices(key, lines):
    (ret1, ret2, ret3, ret4) = (0, 0, 0, 0)
    for i in range(len(lines)):
        position = key[i]
        if position == 'a':
            ret2 += lines[i].getNormal()
            ret4 += lines[i].getNormal()
        if position == 'b':
            ret3 += lines[i].getNormal()
            ret4 += lines[i].getNormal()
        if position == '1':
            ret1 += lines[i].getNormal()
            ret2 += lines[i].getNormal()
            ret3 += lines[i].getNormal()
            ret4 += lines[i].getNormal()            
    return (ret1, ret2, ret3, ret4)


frameCount = 1
height = 6400
width = 6400
length = 80
cGreen = (0, 255, 0)
cLine = (0, 0, 0)
cBackground = (255, 93, 15)
lineCount = 20

def z2imgPoint(z):
    return (width / 2 + int(np.real(z * length)), height / 2 - int(np.imag(z * length)))

def drawRhomboid(img, v1, v2):
    cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = cLine, thickness = 4)

def genLines(ps, f):
    lines = []
    for i in ps:
        l = line(f(i), derivative(f, i))
#       print(l)
        lines.append(l)
    return lines
    
def drawImg(lines):
    img =  np.zeros((height,width,3), np.uint8)
    img[:,:] = cBackground
    faceKeys = genRhomboidFaceKeys(lines)
    for key in faceKeys:
        (v1, v2, v3, v4) = getRhomboidVertices(key, lines)
        drawRhomboid(img, v1, v2)
        drawRhomboid(img, v1, v3)
        drawRhomboid(img, v2, v4)
        drawRhomboid(img, v3, v4)
#    print("k = {} Count of Rhomboids {}".format(k, len(faceKeys)))
    return img

def genK(n):
#    return 2.5 - 5.0 * n / frameCount
    return 4.0 ** (1.0 * (frameCount - 1 - n) / (frameCount - 1)) * 7.0 ** (1.0 * n / (frameCount - 1))

def derivative(f, x):
    h = 0.0001
    return (f(x + h) - f(x)) / h

#ks = [range() for n in range(frameCount)]

ks = [list(np.linspace(-1, 1, lineCount)) for n in range(frameCount)]

angle = (np.sqrt(5) - 1) / 2 * fullAngle * 1j
rotation = lambda x : x + x ** 2 * 1j
rotation = lambda x : np.exp(angle * x)

def curry(z):
    def ret(x):
        angle = (np.sqrt(5) - 1) / 2 * fullAngle * 1j + z * fullAngle * 1j 
        retx =  np.exp(angle * x)
        retx =  x * np.exp(angle * x)
        retx =  (1 - z + x * z) * np.exp(angle * x)
        return retx
    return ret
    
fs = [(lambda x : np.exp(angle * (x + 1.0 * n / frameCount))) for n in range(frameCount)]
fs = [(lambda x : np.exp(angle * (x + 0.1 * n))) for n in range(frameCount)]
fs = [curry(z) for z in list(np.linspace(0, 1, frameCount + 1))][:-1]

print("------------------------------------------------------------")
print ks
print [f(1) for f in fs]

print("Rotation at point 0 = {}".format(rotation(0)))
print("dRotation at point 0 = {}".format(derivative(rotation, 0)))



ims = []
for k, f in zip(ks, fs):
    lines = genLines(k, f)
    print("k[0] = {} f(1) = {} line[0] = {}".format(k[0], f(1), lines[0]))
    img = drawImg(lines)
    img = cv2.resize(img, (1600, 1600))
    if frameCount < 10:
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ims.append(img)


if frameCount > 10:
    imageio.mimwrite(uri = 'rhomboids.mp4', ims = ims)#, duration = 0.05)
#imageio.mimwrite(uri = 'rhomboids.mp4', ims = ims + list(reversed(ims)))#, duration = 0.05)
#imageio.mimwrite(uri = 'rhomboids.gif', ims = ims + list(reversed(ims)), duration = 0.04)

cv2.imwrite('test.jpg', img)



