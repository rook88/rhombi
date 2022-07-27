import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFilter

fullAngle = 2 * np.pi
a2pi = np.pi * 1j
rightAngle = 0.5 * np.pi
origin = 0 + 0j
resolution = None
theta = (np.sqrt(5) - 1) / 2
diskRadius = False
diskOffset = 0
unitIJ = 1 + 1j
   
class img():
    def __init__(self, im = None, resolution = None, size = None, zoom = 1, pipe = False, color = "black"):
        width, height = (480, 480)
        if resolution is not None:
            if resolution == '1080p':
                (width, height) = (1920, 1080)
            if resolution == '720p':
                (width, height) = (1280, 720)
            if 'x' in resolution:
                (width, height) = [int(s) for s in resolution.split('x')]
        if size is not None:
            (width, height) = size
        self.width = width 
        self.height = height
        self.scale = int(max(width, height) / 2)
        if im is not None:
            self.im = im
        else:
            self.im = Image.new("RGB", (width, height), color)
        self.draw = ImageDraw.Draw(self.im)
        self.zoom = zoom
        self.pipe = pipe
    def get(self):
        return self.im
    def getMask(self):
        return self.im.convert("1")
    def z2point(self, zIn, noPipe = False):
        if self.pipe and not noPipe:
            # print(zIn)
            z = np.exp(zIn) / self.zoom 
        else:
            z = zIn / self.zoom
        ret = (int(self.width / 2 + np.real(z) * self.scale), int(self.height / 2 + np.imag(z)  * self.scale))
        # print(ret)
        return ret
    def drawPoint(self, z, radius = 0, color = "white"):
        self.draw.ellipse((self.z2point(z - radius * 0.35 * unitIJ), self.z2point(z + radius * 0.35 * unitIJ)), color)
    def drawEgde(self, z1, z2, color = "white"):
        if abs(z1.imag) < np.pi and abs(z1.imag) > - np.pi:
            self.draw.line((self.z2point(z1), self.z2point(z2)), color)
    def drawLine(self, z1, z2, color = "white", thickness = 1):
        self.draw.line((self.z2point(z1), self.z2point(z2)), color, thickness)
    def drawPolygon(self, zs, color = "white", scale = 1):
        self.draw.polygon([self.z2point(z * scale) for z in zs], fill = color)
    def drawCircle(self, center, radius, color = "white"):
        self.draw.ellipse((self.z2point(center - radius * unitIJ, noPipe = True), self.z2point(center + radius * unitIJ, noPipe = True)), fill = None, outline = color)
    def drawAxes(self):
        if self.pipe:
            self.drawCircle(0, np.exp(np.pi), color = "red")
        else:
            self.drawLine(-self.zoom + np.pi * 1j, self.zoom + np.pi * 1j, color = "green")
            self.drawLine(-self.zoom - np.pi * 1j, self.zoom - np.pi * 1j, color = "green")
            # self.drawLine(-self.zoom, self.zoom)
            # self.drawLine(-self.zoom * 1j, self.zoom * 1j)
            self.drawLine(-self.zoom * 1j + np.pi, self.zoom * 1j + np.pi, color = "red")
            self.drawLine(-self.zoom * 1j - 2 * np.pi, self.zoom * 1j - 2 * np.pi, color = "blue")


class line():
    def __init__(self, somePoint = None, directionPoint = None, normalLength = 0.1, visible = 1.0, angle = None, offset = None):
        if somePoint != None and directionPoint != None:
            self.somePoint = somePoint
            self.direction = directionPoint / abs(directionPoint)
            self.angle = np.imag(np.log(directionPoint)) / 2 / np.pi
            self.normalLength = normalLength
            self.visible = visible
        if angle != None and offset != None:
            self.direction = np.exp(angle * 2j * np.pi) 
            self.somePoint = self.direction * offset / 1j
            self.angle = angle
            self.normalLength = normalLength
            self.visible = visible

    def getNormal(self):
        return self.direction / np.exp(rightAngle * 1j) * self.normalLength * self.visible
    def isVisible(self):
        return self.visible
    def __str__(self):
        return f"Line somePoint = {self.somePoint:+.2} angle = {self.angle / fullAngle:+.2} direction = {self.direction:+.2} normaLength = {self.normalLength:+.2}"


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
    r = 9.0
    distance = np.imag((x - l.somePoint) / l.direction)
    pos = np.exp(-distance * r) / (1 + np.exp(-distance * r)) 
    pos = np.round(pos, 3)
    if distance >= 2:
        return "0", 0
    elif distance <= -2:
        return "1", 1
    elif distance >= 0:
        return "0", pos
    else:
        return "1", pos

def getEmptyImg():
    return np.zeros((imgHeight,imgWidth,3), np.uint8)

class rhombiCl():

    faces = {}
    edges = {} 
    vertices = {}

    @staticmethod
    def genFace(faceKey, faceVector, lines = None, ins = None):
        if faceKey in rhombiCl.faces:
            return rhombiCl.faces[faceKey]
        else:
            return rhombiCl.face(faceKey, faceVector, lines, ins)

    class face():
        def __init__(self, faceKey, faceVector, lines = None, ins = None):
            self.faceKey = faceKey
            self.faceVector = faceVector
            self.lines = lines
            self.index = ins
            e1 = rhombiCl.genEdge(faceKey.replace('a', 'x').replace('b', '0'), self)
            e2 = rhombiCl.genEdge(faceKey.replace('a', 'x').replace('b', '1'), self)
            e3 = rhombiCl.genEdge(faceKey.replace('a', '1').replace('b', 'x'), self)
            e4 = rhombiCl.genEdge(faceKey.replace('a', '0').replace('b', 'x'), self)
            v1 = rhombiCl.genVertice(faceKey.replace('a', '0').replace('b', '0'))
            v2 = rhombiCl.genVertice(faceKey.replace('a', '0').replace('b', '1'))
            v3 = rhombiCl.genVertice(faceKey.replace('a', '1').replace('b', '1'))
            v4 = rhombiCl.genVertice(faceKey.replace('a', '1').replace('b', '0'))
            self.edges = [e1, e2, e3, e4]
            self.vertices = [v1, v2, v3, v4]
            self.center = (v1.position + v3.position) / 2
            rhombiCl.faces[faceKey] = self
        def __str__(self):
            return "face:" + self.faceKey + str([str(v.position) for v in self.vertices])
        def getIndex(self):
            return self.index
        def getDirection(self):
            ret = self.lines[0].getNormal() + self.lines[1].getNormal() 
#            if np.real(ret) < 0:
#                ret = -ret
            return ret
        def getCopy(self):
            ret = type('', (), {})() 
            ret.center = np.array(self.center)
            v1, v2, v3, v4 = self.vertices 
            d1 = v1.position - v3.position
            d2 = v2.position - v4.position
            if abs(d1) > abs(d2):
                ret.vectors = (d1, d2)
                ret.direction = np.imag(np.log(d1))
                ret.angle = abs(np.imag(np.log((v2.position - v1.position) / (v4.position - v1.position))))
            else:
                ret.vectors = (d2, d1)
                ret.direction = np.imag(np.log(d2))
                ret.angle = abs(np.imag(np.log((v1.position - v2.position) / (v3.position - v2.position))))
            ret.color = self.color
            return ret
        def isVisible(self):
            return self.lines[0].isVisible() * self.lines[1].isVisible()
        def isRhombus(self):
            if self.lines[0].normalLength == self.lines[1].normalLength:
                return True
            else:
                return False
        def getShape(self):
            return abs(np.real(self.lines[0].getNormal() / self.lines[1].getNormal() ))
        def setColor(self, color, split, byDirection = 0.0, byFunction = None, onlyRhombus = False):
            h1, s1, v1 = color
            # col1 = cv2.cvtColor(np.uint8([[[int(h1), int(s1), int(v1)]]]), cv2.COLOR_HSV2RGB)
            d = self.getDirection()
            h2 = int(180 * (abs(np.imag(np.log((d))) / 2 / np.pi % 1.0)))
            s2 = 255
            v2 = 255 
            # col2 = cv2.cvtColor(np.uint8([[[int(h2), int(s2), int(v2)]]]), cv2.COLOR_HSV2RGB)
            r1 = col1[0][0][0]
            g1 = col1[0][0][1]
            b1 = col1[0][0][2]
            r2 = col2[0][0][0]
            b2 = col2[0][0][1]
            g2 = col2[0][0][2]
#            print col1, col2, r1, g2, b1, r2, g2, b2
            """* self.isVisible()
                if hue != None:
                    H = int(hue * saturation / 255.0 + (1 - saturation / 255.0) * 180 (abs(np.imag(np.log((d))) / np.pi % 1.0)))
                else:
            """
            r = r1 * (1 - byDirection) + r2 * byDirection
            g = g1 * (1 - byDirection) + g2 * byDirection
            b = b1 * (1 - byDirection) + b2 * byDirection
            h = h1 * (1 - byDirection) + h2 * byDirection
            s = s1 * (1 - byDirection) + s2 * byDirection
            v = v1 * (1 - byDirection) + v2 * byDirection
            # col3 = cv2.cvtColor(np.uint8([[[int(r), int(g), int(b)]]]), cv2.COLOR_RGB2HSV)
            h = float(col3[0][0][0])
            s = float(col3[0][0][1])
            v = int(col3[0][0][2])
            if byFunction:
                (h, s, v) = byFunction(self.faceKey)
            if onlyRhombus:
                if not self.isRhombus():
                    (h, s, v) = (0, 0, 50)
            self.color = (h, s, v)
            self.split = split
        def draw(self, img, edgeColor = None, offset = 0, scale = 1, color = "white"):
            vertices = self.vertices
            positions = [v.getPosition() for v in vertices]
            innerPositions = [v.getPosition() * 0.95 + 0.05 * self.center for v in vertices]
            # h, s, v = self.color
            # vs = (255 * self.split + (1 - self.split) * v) * self.isVisible()
            # v = int(v * self.isVisible())
#            print h, s, v
            if edgeColor is not None:
                img.drawPolygon(positions, color = edgeColor, scale = scale)
                img.drawPolygon(innerPositions, color = color, scale = scale)
            else:
                img.drawPolygon(positions, color = color, scale = scale)
        def drawEdges(self, img, edgeColor, offset = 0, scale = 1):
            for edge in self.edges:
                edge.draw(img, offset = offset, scale = scale, color = edgeColor)
        
        def drawSmooth(self, img):
            p = 0.0
            for i in range(len(rhombiCl.lines)):
                n = rhombiCl.lines[i].getNormal()
                if self.faceVector[i] == 'a':
                    posA = n
                elif self.faceVector[i] == 'b':
                    posB = n
                else:
                    p += n * self.faceVector[i]
            positions = [p, p + posA, p + posA + posB, p + posB]
#            print self.faceVector
            points = [z2imgPoint(pos) for pos in positions]
            # cv2.fillConvexPoly(img, np.array(points), self.color)
            # cv2.line(img = img, pt1 = points[0], pt2 = points[1], color = (0,0,0), thickness = 20)
            # cv2.line(img = img, pt1 = points[1], pt2 = points[2], color = (0,0,0), thickness = 20)
            # cv2.line(img = img, pt1 = points[2], pt2 = points[3], color = (0,0,0), thickness = 20)
            # cv2.line(img = img, pt1 = points[3], pt2 = points[0], color = (0,0,0), thickness = 20)
            

    @staticmethod
    def genEdge(edgeKey, face):
        if edgeKey in rhombiCl.edges:
            e = rhombiCl.edges[edgeKey]
            e.faces.append(face)
            return e 
        else:
            ret = rhombiCl.edge(edgeKey, face)
            rhombiCl.edges[edgeKey] = ret
            return ret

    class edge():
        def __init__(self, edgeKey, face):
            self.edgeKey = edgeKey
            self.faces = [face]
            v1 = rhombiCl.genVertice(edgeKey.replace('x', '0'))
            v2 = rhombiCl.genVertice(edgeKey.replace('x', '1'))
            self.vertices = [v1, v2]
            self.split = False
            rhombiCl.edges[edgeKey] = self
        def __str__(self):
            return self.edgeKey + " " + str(self.split)
        def setColor(self, color, thickness):
            if thickness == None:
                self.thickness = 10
            else:
                self.thickness = int(thickness)
            if self.split:
                self.color = self.faces[0].color
            elif color and len(self.faces) == 2:
                (h, s, v) = color
                d1 = self.faces[0].getDirection()
                d2 = self.faces[1].getDirection()
                dif = abs(d1 - d2) * abs(d1 + d2) / abs(d1) / abs(d2)
                dif = int(100 * dif)
                v = np.clip(dif, 0, 255)
                self.color = [h, s, v]
                self.color = color
            elif color:
                self.color = color
            elif len(self.faces) == 2:
                """
                d1 = self.faces[0].getDirection()
                d2 = self.faces[1].getDirection()
                dif = abs(d1 - d2) * abs(d1 + d2) / d1 / d2
                hue = int(180 * (abs(np.imag(np.log((d1))) / np.pi % 1.0)))
                value = 255
                value = 255 - min(255, int(dif * 128))
                self.color = [hue, 255, value]
                """
                self.color = (0, 0, 0)
            else:
                self.color = None
        def draw(self, img, offset = 0, scale = 1 + 0j, color = "white"):
#            imgTemp = getEmptyImg()
            # if self.color != None:
            v1, v2 = self.vertices
            # print(v1.position, v2.position)
            img.drawLine(v1.getPosition(scale) + offset, v2.getPosition(scale) + offset, color = color, thickness = 2)
                # cv2.line(img = img, pt1 = pt1, pt2 = pt2, color = self.color, thickness = self.thickness)
#                img = cv2.add(img, imgTemp)

    @staticmethod
    def genVertice(verticeKey):
        if verticeKey in rhombiCl.vertices:
            return rhombiCl.vertices[verticeKey]
        else:
            return rhombiCl.vertice(verticeKey)

    class vertice():
        def __init__(self, verticeKey):
            self.verticeKey = verticeKey
            position = 0
            for i in range(len(rhombiCl.lines)):
                if verticeKey[i] == '1':
                    position += rhombiCl.lines[i].getNormal()
            self.position = position
            rhombiCl.vertices[verticeKey] = self
        def __str__(self):
            return 'vertice:' + self.verticeKey
        def setColor(self, color, radius):
            self.color = color
            self.radius = int(radius)
        def asPoint(self, zoom):
            return z2imgPoint(self.position, zoom = zoom)
        def draw(self, img):
            pt = z2imgPoint(self.position)
            # cv2.circle(img, pt, self.radius, self.color, -1)
        def getPosition(self, scale = 1 + 0j):
            # print(type(self.position), self.position, scale)
            return scale * self.position

    def __init__(self, lines):

        rhombiCl.lines = lines
        rhombiCl.faces.clear()
        rhombiCl.edges.clear()
        rhombiCl.vertices.clear()
        previousIntersections = []
        for i1 in range(len(lines)):
            for i2 in range(i1 + 1, len(lines)):
                p = intersection(lines[i1], lines[i2])
                if p != None:
                    if p in previousIntersections:
                        raise ValueError("multiple lines intersect at same point", p)
                    previousIntersections.append(p)
                    faceKey = ""
                    faceVector = []
                    for i3 in range(len(lines)):
                        if i3 == i1:
                            faceKey += "a"
                            faceVector.append("a")
                        elif i3 == i2:
                            faceKey += "b"
                            faceVector.append("b")
                        else:
                            posStr, pos = position(lines[i3], p)
                            faceKey += posStr
                            faceVector.append(pos)
#                    print faceKey, faceVector
                    f = rhombiCl.genFace(faceKey, faceVector, (lines[i1], lines[i2]), (i1, i2))
    
        self.faces = rhombiCl.faces
        self.edges = rhombiCl.edges
        self.vertices = rhombiCl.vertices
    
    def get(self):
        return copy.deepcopy(self)

    def setColors(self, hue = None, saturation = None, value = None, faceColor = None, faceSplit = 0.0, faceByDirection = 0.0, faceByFunction = None, edgeColor = None, edgeThickness = None, verticeRadius = 10, faceOnlyRhombus = False):
        if not saturation:
            saturation = 255
        if not edgeColor:
            edgeColor = (hue, saturation, value)
        if not faceColor:
            faceColor = (hue, saturation , value)
        for faceKey, face in self.faces.items():
            face.setColor(color = faceColor, split = faceSplit, byDirection = faceByDirection, byFunction = faceByFunction, onlyRhombus = faceOnlyRhombus)
        for edgeKey, edge in self.edges.items():
            edge.setColor(color = edgeColor, thickness = edgeThickness)
        for verticeKey, vertice in self.vertices.items():
            vertice.setColor(color = edgeColor, radius = verticeRadius)


    def getImg(self, smooth = False):
        img =  getEmptyImg()
        img[:,:] = backgroundColor
        for faceKey, face in self.faces.items():
            if smooth:
                face.drawSmooth(img)
            else:
                face.draw(img)
        for edgeKey, edge in self.edges.items():
            if not smooth:
                edge.draw(img)
        for verticeKey, vertice in self.vertices.items():
            if not smooth:
                vertice.draw(img)
        return img

    def __str__(self):
        return f"Rhombi, faces = {len(self.faces)}, edges = {len(self.edges)}, vertices = {len(self.vertices)}"


def z2imgPoint(z, diskRadius = False, zoom = 1.0):
    if diskRadius:
        z = z + diskOffset
        r = abs(z)
        d = z / r
        r = r / diskRadius
        r = r / np.sqrt(r*r + 1)
        z = r * d
        x = np.real(z)
        y = np.imag(z)
        return (int(imgWidth / 2 * (x + 1)), int(imgHeight / 2 * (y + 1)))
    else:
        return (int(imgWidth / 2 + (np.real(z * zoom))), int(imgHeight / 2 - np.imag(z * zoom)))

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


def drawLines(im, lines, length = 3000, color = "white", thickness = 1):
    for line in lines:
        im.drawLine(line.somePoint - length * line.direction, line.somePoint + length * line.direction, color = color, thickness = thickness)

def genLines(radii, angles, center = 0):
    ret = []
    for n in range(len(radii)):
        a = angles[n] * a2pi
        sp = np.exp(a) * radii[n] + center
        d = 1j * np.exp(a)
        l = line(sp, d, 1.0, 1.0)
        ret.append(l)
    return ret
        
def genStar(lineCount, angle = (np.sqrt(5) - 1) / 2, angleDelta = 0.0, radiusMin = 0.0, radiusMax = 1.0, center = 0.0, time = None, normalLength = 1.0, step = 1, radiusDelta = None, radii = None):
    if time != None:
        angleDelta *= time
    ret = []
    length = normalLength
    visible = 1.0
    for n in range(int(np.ceil(lineCount))):
#    for n in range(int(lineCount)):
        a = (angle * n + angleDelta) * a2pi
#         print(angle * n)
        nrad = n - n % step
#        r = (radiusMin * n + radiusMax * (lineCount - n)) / lineCount
        if radiusDelta is None and radii is None:
            r = (radiusMin * nrad + radiusMax * (lineCount - nrad)) / lineCount
        elif radii is not None:
            r = radii[n]
        else:
            r = (n * radiusDelta) % 1 - 0.5
        sp = np.exp(a) * r + center
        d = 1j * np.exp(a)
        if n + 1 > lineCount:
            visible = (lineCount - n)
        l = line(sp, d, length, visible)
        ret.append(l)
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
    return min(ret * nCount * np.sqrt(nCount), 0.20) * 5

    
