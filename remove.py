import os, glob
import cv2
from multiprocessing import Pool
from PIL import Image

files = glob.glob(os.path.join('Photos','train_1','*.jpg'))

def remove((i,file)):
    try:
        # bgr_img = cv2.imread(file, -1)
        # try:
        #     b, g, r = cv2.split(bgr_img)  # get b,g,r
        # except:
        #     print(file)
        #     os.remove(file)
        im = Image.open(file)
        im.verify()
    except:
        os.remove(file)
        print(file)

    print(i)

if __name__ == '__main__':
    # p = Pool(6)
    dat = []
    files.sort()
    for i, file in enumerate(files, 1):
        bgr_img = cv2.imread(file, -1)
        try:
            b, g, r = cv2.split(bgr_img)  # get b,g,r
        except:
            print(file)
            os.remove(file)
        print(i, file)
        # dat.append((i, file))

    # print(p.map(remove, dat))