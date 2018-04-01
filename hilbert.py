import numpy as np

def hRandom(seed):
    theta = (-1 + np.sqrt(5)) / 2
    z = theta * (seed + 1) % 1
    return i2hClosed(z)

def hOrder(i, j, n, im = 0):
    if im == 0:
        m = n
    else:
        m = im
    x = hRound(i, n)
    y = hRound(j, m)
    ret = h2i(x, y)
    return ret

def hRound(k, n, m = 1):
    if n < 2:
        return 0.25
    if n == 2:
        if k == 1:
            return 0.25 / m
        else:
            return 1 - 0.25 / m
    n2 = (n + 1) / 2
    if k <= n2:
        return 0.5 * hRound(k, n2)
    else:
        return 1.0 - 0.5 * hRound(n + 1 - k, n - n2, (n % 2) + 1)

        

def d2xy(d):
    if d < 2:
        return (d, 0)
    i = int(np.floor(np.log(d) / np.log(2)))
    p2 = 2 ** i
    p1 = 2 ** (i / 2)
#    x, y = d % 2, (d / 2) % 2
    direction = ((i + 1) % 4) / 2  # 0 = x, 1 = y
    if i % 2 == 1: # mirroring
        xp, yp = d2xy(p2 * 2 - 1 - d)
        if direction == 0:
            (x, y) =  (p1 * 2 - 1 - xp, yp) 
        else:
            (x, y) = (xp, p1 * 2 - 1 - yp) 
    else: # diagonal mirroring and transform
        xp, yp = d2xy(d - p2)
        if direction == 0:
            (x, y) = (yp + p1, xp)
        else:
            (x, y) = (yp, xp + p1)
#    print "d={}, i={}, p1={}, p2={}, xp={}, yp={}, d={}".format(d, i, p1, p2, xp, yp, direction)
    return (x, y)

def h2i(ix, iy, level = 1.0):
    if level < accuracy:
        return 1.0 / 3
    x = ix * 2
    if x > 1:
        x -= 1
    y = iy * 2
    if y > 1:
        y -= 1
    if ix <= 0.5 and iy <= 0.5:
        ret = 0.0 + h2i(y, x, level / 4) / 4
    elif ix <= 0.5 and iy  > 0.5:
        ret = 0.25 + h2i(x, y, level / 4) / 4
    elif ix  >= 0.5 and iy > 0.5:
        ret = 0.5 + h2i(x, y, level / 4) / 4
    else:
        ret = 0.75 + h2i(1 - y, 1 - x, level / 4) / 4
    return ret


def i2h(z, level = 1.0):
    if level < accuracy:
        return 0.5, 0.5
    t = int(z * 4)
    z2 = z * 4 - t
    ix, iy = i2h(z2, level / 4)
    x, y = ix / 2, iy / 2
    if t == 0:
        return y, x
    elif t == 1:
        return x, 0.5 + y
    elif t == 2:
        return 0.5 + x, 0.5 + y
    else:
        return 1.0 - y, 0.5 - x


def i2hClosedRound(z, round = 0.0):
    retx = 0
    rety = 0
    for i in range(3):
        x, y = i2hClosed(z + i * round)
        retx += x
        rety += y
        x, y = i2hClosed(z - i * round)
        retx += x
        rety += y
    return retx / 6, rety / 6

accuracy = 2.0 ** -20.0

def i2hClosed(z, level = 1.0):
    if level < accuracy:
        return 0.5, 0.5
    t = int(z * 4)
    z2 = z * 4 - t
    ix, iy = i2h(z2, level / 4)
    x, y = ix / 2, iy / 2
    if t == 0:
        return 0.5 - y, x
    elif t == 1:
        return 0.5 - y, 0.5 + x
    elif t == 2:
        return 0.5 + y, 1.0 - x
    else:
        return 0.5 + y, 0.5 - x

