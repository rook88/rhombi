import imageio

ims = []


frameCount = 0

def appendIms(fileName):
    global frameCount
    reader = imageio.get_reader(fileName)
    for i, im in enumerate(reader):
        ims.append(im)
        frameCount += 1
        print frameCount

appendIms('slide01.mp4')
appendIms('slide02.mp4')

print "writing..."
imageio.mimwrite(uri = 'slideAll.mp4', ims = ims, macro_block_size = None, fps = 24)
