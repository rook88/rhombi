import numpy as np
import cv2
import getopt, sys


fullAngle = 2 * np.pi
a2pi = 2 * np.pi * 1j
rightAngle = 0.5 * np.pi
origin = 0 + 0j

frameCount = 3
lineCount = 20
edgeLength = 25 * 5
testMode = False
outputFile = None
testT = None
testAngle = None
testAngleMin = None
testAngleMax = None
testLineCount = None
testFrameCount = None
resolution = None

opts, args = getopt.getopt(sys.argv[1:], None, ['testmode', 'file=', 'angle=','anglemin=','anglemax=', 'linecount=', 'framecount=', 'testtime=', 'resolution='])

for o, a in opts:
    if o == '--testmode':
        testMode = True
    if o == '--file':
        outputFile = a
    if o == '--testtime':
        testT = float(a)
    if o == '--angle':
        testAngle = float(a)
    if o == '--anglemin':
        testAngleMin = float(a)
    if o == '--anglemax':
        testAngleMax = float(a)
    if o == '--linecount':
        testLineCount = int(a)
    if o == '--resolution':
        resolution = a
    if o == '--framecount':
        testFrameCount = int(a)
        
(width, height) = (640, 480)
if resolution:
    if resolution == '1080p':
        (width, height) = (1920, 1080)
    if resolution == '720p':
        (width, height) = (1280, 720)

if testLineCount:
    lineCount = testLineCount

imgWidth = width * 5
imgHeight= height * 5
backgroundColor = (0, 0, 0)
rhombusEdgeThickness = 10

theta = (np.sqrt(5) - 1) / 2
if testAngleMin:
    angleMin = testAngleMin
else:
    angleMin = theta

if testAngleMax:
    angleMax = testAngleMax
else:
    angleMax = 1 - (1 - angleMin) / 2


ts = list(np.linspace(0, 1, frameCount))
if testT:
    ts = [testT]
    frameCount = 1

if testAngle:
    t = (testAngle - angleMin) / (angleMax - angleMin)
    ts = [t]
    print("t = {}".format(t)) 
    frameCount = 1

if testFrameCount:
    frameCount = testFrameCount

class line():
    def __init__(self, somePoint = None, directionPoint = None, normalLength = 1.0, visible = 1.0):
        if somePoint <> None and directionPoint <> None:
            self.somePoint = somePoint
            self.direction = directionPoint / abs(directionPoint)
            self.angle = np.imag(np.log(directionPoint))
            self.normalLength = normalLength
            self.visible = visible
    def getNormal(self):
        return self.direction / np.exp(rightAngle * 1j) * self.normalLength * self.visible
    def isVisible(self):
        return self.visible
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
    
def genRhombiFaceKeys(lines):
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



faceLines = {}
faceEdges = {} 
faceVertices = {} 
faceDirections = {} 
faceVisibles = {} 
faceShapes = {} 
edgeFaces = {} 
edgeVertices = {} 
vertices = {}
def genRhombi(lines):
    faceEdges.clear()
    edgeFaces.clear()
    edgeVertices.clear()
    faceVertices.clear()
    faceLines.clear()
    faceDirections.clear()
    faceShapes.clear()
    faceVisibles.clear()
    previousIntersections = []
    for i1 in range(len(lines)):
        for i2 in range(i1 + 1, len(lines)):
            p = intersection(lines[i1], lines[i2])
            if p <> None:
                if p in previousIntersections:
                    raise ValueError("multiple lines intersect at same point", p)
                previousIntersections.append(p)
                faceKey = ""
                for i3 in range(len(lines)):
                    if i3 == i1:
                        faceKey += "a"
                    elif i3 == i2:
                        faceKey += "b"
                    else:
                        faceKey += str(position(lines[i3], p))
                faceLines[faceKey] = (i1, i2)
                faceDirections[faceKey] = lines[i1].getNormal() + lines[i2].getNormal() 
                faceShapes[faceKey] = lines[i1].getNormal() / lines[i2].getNormal() 
                faceVisibles[faceKey] = lines[i1].visible * lines[i2].visible
    for faceKey in faceLines:
        edgeKey1 = ""
        edgeKey2 = ""
        edgeKey3 = ""
        edgeKey4 = ""
        for i in range(len(lines)):
            pos = faceKey[i]
            if pos == 'a':
                edgeKey1 += 'x'
                edgeKey2 += 'x'
                edgeKey3 += '0'
                edgeKey4 += '1'
            elif pos == 'b':
                edgeKey1 += '0'
                edgeKey2 += '1'
                edgeKey3 += 'x'
                edgeKey4 += 'x'
            else:
                edgeKey1 += pos
                edgeKey2 += pos
                edgeKey3 += pos
                edgeKey4 += pos
        faceEdges[faceKey] = [edgeKey1, edgeKey2, edgeKey3, edgeKey4]
        edgeFacesAppend(edgeKey1, faceKey)
        edgeFacesAppend(edgeKey2, faceKey)
        edgeFacesAppend(edgeKey3, faceKey)
        edgeFacesAppend(edgeKey4, faceKey)
        faceVertices[faceKey] = getEdgeVertices(edgeKey1, lines) + list(reversed(getEdgeVertices(edgeKey2, lines)))
    for edgeKey in edgeFaces:
        edgeVertices[edgeKey] = getEdgeVertices(edgeKey, lines)

        
def edgeFacesAppend(edgeKey, faceKey):
    if edgeFaces.has_key(edgeKey):
        edgeFaces[edgeKey].append(faceKey)
    else:
        edgeFaces[edgeKey] = [faceKey]

def getEdgeVertices(key, lines):
    (ret1, ret2) = (0, 0)
    for i in range(len(lines)):
        pos = key[i]
        if pos == 'x':
            ret1 += lines[i].getNormal()
        if pos == '1':
            ret1 += lines[i].getNormal()
            ret2 += lines[i].getNormal()
    return [ret1, ret2]



"""
                faceShapes[faceKey] = lines[i1].getNormal() / lines[i2].getNormal() 
                faceVisibles[faceKey] = lines[i1].visible * lines[i2].visible
"""


class rhombi():

    faces = {}
    edges = {} 
    vertices = {}

    class face():
        def __init__(self, faceKey, lines = None):
            if faceKey in rhombi.faces:
                self = rhombi.faces[faceKey]
            else:
                self.faceKey = faceKey
                self.lines = lines
                rhombi.faces[faceKey] = self
        def getDirection(self):
            return self.lines[0].getNormal() + self.lines[1].getNormal() 
        def isVisible(self):
            return self.lines[0].getNormal() + self.lines[1].getNormal() 

    class edge():
        def __init__(self, edgeKey, face):
            if edgeKey in rhombi.edges:
                selfD = rhombi.edges[edgeKey]
                selfD.faces.append(face)
                self =  selfD
            else:
                self.edgeKey = edgeKey
                self.faces = [face]
                rhombi.edges[edgeKey] = self

    def __init__(self, lines):

        previousIntersections = []
        for i1 in range(len(lines)):
            for i2 in range(i1 + 1, len(lines)):
                p = intersection(lines[i1], lines[i2])
                if p <> None:
                    if p in previousIntersections:
                        raise ValueError("multiple lines intersect at same point", p)
                    previousIntersections.append(p)
                    faceKey = ""
                    for i3 in range(len(lines)):
                        if i3 == i1:
                            faceKey += "a"
                        elif i3 == i2:
                            faceKey += "b"
                        else:
                            faceKey += str(position(lines[i3], p))
                    f = rhombi.face(faceKey, (lines[i1], lines[i2]))
                    e1 = rhombi.edge(faceKey.replace('a', 'x').replace('b', '0'), f)
                    e2 = rhombi.edge(faceKey.replace('a', 'x').replace('b', '1'), f)
                    e3 = rhombi.edge(faceKey.replace('a', '1').replace('b', 'x'), f)
                    e4 = rhombi.edge(faceKey.replace('a', '0').replace('b', 'x'), f)
                    f.edges = [e1, e2, e3, e4]
#                    f.vertices = getEdgeVertices(edgeKey1, lines) + list(reversed(getEdgeVertices(edgeKey2, lines)))
    
        for k, e in rhombi.edges.items():
            e.vertices = getEdgeVertices(k, lines)
    
        self.faces = rhombi.faces
        self.edges = rhombi.edges
        self.vertices = rhombi.vertices
    
    def getImg(self, edgeColor = (0, 255, 0)):
        img =  np.zeros((imgHeight,imgWidth,3), np.uint8)
        img[:,:] = backgroundColor
#        for faceKey, face in faces.items():
#            drawRhombus(img, faceKey)
        for edgeKey, edge in self.edges.items():
            drawEdgeNew(img, edge, edgeColor)
        return img

    def __str__(self):
        return "Rhombi, faces = {}, edges = {}".format(len(self.faces), len(self.edges))

def z2imgPoint(z):
    return (imgWidth / 2 + int(np.real(z)), imgHeight / 2 - int(np.imag(z)))


def drawRhombus(img, faceKey):
    saturation = rhombusFaceColor1[1]
    value = 255 * faceVisibles[faceKey]
    if saturation > 1128:
        vertices1 = faceVertices[faceKey][0:-1]
        points1 = np.array([z2imgPoint(v) for v in vertices1])
        cv2.fillConvexPoly(img, points1, rhombusFaceColor1)
        vertices2 = faceVertices[faceKey][2:] + faceVertices[faceKey][0:1]
        points2 = np.array([z2imgPoint(v) for v in vertices2])
        cv2.fillConvexPoly(img, points2, rhombusFaceColor2)
    else:
        d = faceDirections[faceKey]
#        d = faceShapes[faceKey]
        hue = int(180 * (abs(np.imag(np.log((d))) / np.pi % 1.0)))
#        hue = (int(180 * (abs(np.real(d)))) + 105) % 180
#        color = (hue, 255 - saturation, value)
        color = (hue, saturation, value)
        vertices = faceVertices[faceKey]
        points = np.array([z2imgPoint(v) for v in vertices])
        cv2.fillConvexPoly(img, points, color)
                  

def drawEdge(img, edgeKey):
    (v1, v2) = edgeVertices[edgeKey]
    if len(edgeFaces[edgeKey]) == 2:
        d1 = faceDirections[edgeFaces[edgeKey][0]]
        d2 = faceDirections[edgeFaces[edgeKey][1]]
        dif = abs(d1 - d1) * abs(d1 + d2)
        value = min(255, int(dif * 128))
        value = 0
        cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = [rhombusEdgeColor[0], rhombusEdgeColor[1], value], thickness = rhombusEdgeThickness)
    else:
        pass
#        cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = rhombusEdgeColor, thickness = rhombusEdgeThickness * 1)

def drawEdgeNew(img, edge, edgeColor = None):
    (v1, v2) = edge.vertices
    if edgeColor:
#        print v1, v2, edgeColor, z2imgPoint(v1), z2imgPoint(v2)
        cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = edgeColor, thickness = rhombusEdgeThickness)
    elif len(edge.faces) == 2:
        d1 = edge.faces[0].getDirection()
        d2 = edge.faces[1].getDirection()
        dif = abs(d1 - d1) * abs(d1 + d2)
#        value = min(255, int(dif * 128))
        value = 0
        cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = [rhombusEdgeColor[0], rhombusEdgeColor[1], value], thickness = rhombusEdgeThickness)
    else:
        cv2.line(img = img, pt1 = z2imgPoint(v1), pt2 = z2imgPoint(v2), color = (0, 0, 0), thickness = rhombusEdgeThickness)




def genLines(ps, f, g = None):
    lines = []
    for i in ps:
        if g == None:
            l = line(f(i), derivative(f, i))
        else:
            l = line(f(i), g(i))
#       print(l)
        lines.append(l)
    return lines
    

def drawImg(lines):
    img =  np.zeros((imgHeight,imgWidth,3), np.uint8)
    img[:,:] = backgroundColor
    faceKeys = genRhombiFaceKeys(lines)
    for faceKey in faceKeys:
        drawRhombus(img, faceKey)
    for edgeKey in edgeFaces:
        drawEdge(img, edgeKey)

    return img


def drawLines(lines):
    img =  np.zeros((imgHeight,imgWidth,3), np.uint8)
    img[:,:] = backgroundColor
    for line in lines:
        pt1 = line.somePoint - 30 * line.direction
        pt2 = line.somePoint + 30 * line.direction
        drawEdgeOld(img, pt1, pt2)
    return img

def genK(n):
#    return 2.5 - 5.0 * n / frameCount
    return 4.0 ** (1.0 * (frameCount - 1 - n) / (frameCount - 1)) * 7.0 ** (1.0 * n / (frameCount - 1))


def genStar(lineCount, angle = (np.sqrt(5) - 1) / 2, angleDelta = 0.0, radiusMin = 0.0, radiusMax = 1.0, center = 0.0, time = 0.0, doublePct = 0.0, normalLength = 1.0):
    ret = []
    if not doublePct:
        l = normalLength
    else:
        l = normalLength / 2
    for nn in range(lineCount):
        n = nn
        a = (angle * n + angleDelta * time) * a2pi
        r = (radiusMin * n + radiusMax * (lineCount - n)) / lineCount
        sp = np.exp(a) * r + center
        d = 1j * np.exp(a)
        line1 = line(sp, d, l)
        ret.append(line1)
        if doublePct:
            n = nn + doublePct / 2
            a = (angle * n + angleDelta * time) * a2pi
            r = (radiusMin * n + radiusMax * (lineCount - n)) / lineCount
            sp = np.exp(a) * r + center
            d = 1j * np.exp(a)
            line2 = line(sp, d, l)
            ret.append(line2)
    return ret


def measureIrrationality(x):
    nCount = 20
    sx = [ n * x % 1 for n in range(nCount)]
    sx.sort()
    dx = [(sx[n] - sx[n + 1]) ** 2 for n in range(nCount - 1)]
    dx.sort()
    ret = 0
    for i in range(int(np.sqrt(nCount))):
        ret += dx[i]
    return min(ret * nCount * np.sqrt(nCount), 0.10) * 10

    
