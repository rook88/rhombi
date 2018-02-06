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

opts, args = getopt.getopt(sys.argv[1:], None, ['testmode', 'file=', 'angle=','anglemin=','anglemax=', 'linecount=', 'framecount=', 'testtime=', 'resolution=', 'edgelength='])

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
    if o == '--edgelength':
        edgeLength = int(a)
        
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

if testFrameCount:
    frameCount = testFrameCount

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
                e1 = rhombi.edge(faceKey.replace('a', 'x').replace('b', '0'), self)
                e2 = rhombi.edge(faceKey.replace('a', 'x').replace('b', '1'), self)
                e3 = rhombi.edge(faceKey.replace('a', '1').replace('b', 'x'), self)
                e4 = rhombi.edge(faceKey.replace('a', '0').replace('b', 'x'), self)
                v1 = rhombi.vertice(faceKey.replace('a', '0').replace('b', '0'))
                v2 = rhombi.vertice(faceKey.replace('a', '0').replace('b', '1'))
                v3 = rhombi.vertice(faceKey.replace('a', '1').replace('b', '1'))
                v4 = rhombi.vertice(faceKey.replace('a', '1').replace('b', '0'))
                self.edges = [e1, e2, e3, e4]
                self.vertices = [v1, v2, v3, v4]
#  
                rhombi.faces[faceKey] = self
        def __str__(self):
            return "face:" + self.faceKey + str([str(v.position) for v in self.vertices])
        def getDirection(self):
            return self.lines[0].getNormal() + self.lines[1].getNormal() 
        def isVisible(self):
            return self.lines[0].isVisible() * self.lines[1].isVisible() 
        def getShape(self):
            return lines[0].getNormal() / lines[1].getNormal() 
        def draw(self, img, faceColor = None, faceSplit = None, hue = None, saturation = None):
            if faceColor:
                color = faceColor
            else:
                d = face.getDirection()
                if saturation <> None:
                    S = saturation
                else:
                    S = 255
                if hue <> None:
                    H = int(hue * saturation / 255.0 + (1 - saturation / 255.0) * 180 * (abs(np.imag(np.log((d))) / np.pi % 1.0)))
                else:
                    H = int(180 * (abs(np.imag(np.log((d))) / np.pi % 1.0)))
                value = 255 * face.isVisible()
                color = (H, S, value)
            vertices = self.vertices
            points = [z2imgPoint(v.position) for v in vertices]
            points1 = points[0:-1] 
            points2 = points[2:] + points[0:1]
            cv2.fillConvexPoly(img, np.array(points1), color)
            if faceSplit:
                color = color[0], color[1], 255 * faceSplit + (1 - faceSplit) * color[2]
            cv2.fillConvexPoly(img, np.array(points2), color)


    class edge():
        def __init__(self, edgeKey, face):
            if edgeKey in rhombi.edges:
                selfD = rhombi.edges[edgeKey]
                selfD.faces.append(face)
                self = selfD
            else:
                self.edgeKey = edgeKey
                self.faces = [face]
                v1 = rhombi.vertice(edgeKey.replace('x', '0'))
                v2 = rhombi.vertice(edgeKey.replace('x', '1'))
                self.vertices = [v1, v2]
                rhombi.edges[edgeKey] = self
        def draw(self, img, edgeColor = None, hue = None, saturation = None):
            v1, v2 = self.vertices
            pt1, pt2 = z2imgPoint(v1.position), z2imgPoint(v2.position)
            if edgeColor:
#        print v1, v2, edgeColor, z2imgPoint(v1), z2imgPoint(v2)
                color = edgeColor
            elif len(edge.faces) == 2:
                d1 = edge.faces[0].getDirection()
                d2 = edge.faces[1].getDirection()
                dif = abs(d1 - d2) * abs(d1 + d2) / d1 / d2
                hue = int(180 * (abs(np.imag(np.log((d1))) / np.pi % 1.0)))
                value = 255
                value = 255 - min(255, int(dif * 128))
                color = [hue, 255, value]
#        print color, dif, d1, d2
            else:
                return
            cv2.line(img = img, pt1 = pt1, pt2 = pt2, color = color, thickness = rhombusEdgeThickness)


    class vertice():
        def __init__(self, verticeKey):
            if verticeKey in rhombi.vertices:
                selfD = rhombi.vertices[verticeKey]
                self.verticeKey = selfD.verticeKey
                self.position = selfD.position
            else:
                self.verticeKey = verticeKey
                position = 0
                for i in range(len(rhombi.lines)):
                    if verticeKey[i] == '1':
                        position += rhombi.lines[i].getNormal()
                self.position = position
                rhombi.vertices[verticeKey] = self
        def __str__(self):
            return 'vertice:' + self.verticeKey
        def draw(self, img, radius = 0, color = (0, 0, 0)):
            pt = z2imgPoint(self.position)
            cv2.circle(img, pt, radius, color, -1)

    def __init__(self, lines):

        rhombi.lines = lines
        rhombi.faces.clear()
        rhombi.edges.clear()
        rhombi.vertices.clear()
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
    
        self.faces = rhombi.faces
        self.edges = rhombi.edges
        self.vertices = rhombi.vertices
    
    def getImg(self, hue = None, saturation = None, edgeColor = None, faceColor = None, faceSplit = None, verticeRadius = None):
        img =  np.zeros((imgHeight,imgWidth,3), np.uint8)
        img[:,:] = backgroundColor
        for faceKey, face in self.faces.items():
            face.draw(img, faceColor = faceColor, faceSplit = faceSplit, hue = hue, saturation = saturation)
        for edgeKey, edge in self.edges.items():
            edge.draw(img, edgeColor, hue, saturation)
        for verticeKey, vertice in self.vertices.items():
            vertice.draw(img, color = edgeColor, radius = verticeRadius)
        return img

    def __str__(self):
        return "Rhombi, faces = {}, edges = {}, vertices = {}".format(len(self.faces), len(self.edges), len(self.vertices))

def z2imgPoint(z):
    return (imgWidth / 2 + int(np.real(z)), imgHeight / 2 - int(np.imag(z)))

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

    
