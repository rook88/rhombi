import numpy as np
import cv2



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


def z2imgPoint(z):
    return (imgWidth / 2 + int(np.real(z * sideLength)), imgHeight / 2 - int(np.imag(z * sideLength)))

def drawRhomboid(img, v1, v2):
    cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = rhombusEdgeColor, thickness = rhombusEdgeThickness)

def genLines(ps, f):
    lines = []
    for i in ps:
        l = line(f(i), derivative(f, i))
#       print(l)
        lines.append(l)
    return lines
    
def drawImg(lines):
    img =  np.zeros((imgHeight,imgWidth,3), np.uint8)
    img[:,:] = backgroundColor
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

def star(sideCount, center = 0):
    def ret(x):
        angle = fullAngle * 1j / sideCount
        retx =  np.exp(angle * x) + x * center * np.exp(angle * x) / sideCount
        return retx
    return ret

def starNew(z, center = 0):
    def ret(x):
        angle = z * fullAngle * 1j 
        retx =  np.exp(angle * x) + x * center * np.exp(angle * x) 
        return retx
    return ret

def curry(z):
    def ret(x):
        angle = (np.sqrt(5) - 1) / 2 * fullAngle * 1j + z / 10 * fullAngle * 1j 
        retx =  x * np.exp(angle * x)
        retx =  (1 - z + x * z) * np.exp(angle * x)
        retx =  np.exp(angle * x)
        return retx
    return ret

def curryOld(z):
    def ret(x):
        angle = (np.sqrt(5) - 1) / 2 * fullAngle * 1j + 5 * z * fullAngle * 1j 
        return np.exp(angle * x)
    return ret
    



