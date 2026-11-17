import glob
import os

imglist = glob.glob('/media/penglong/致钛2T/PL/data/make3d/Test134/*.jpg')

with open('make3d_test_files.txt', 'a+') as f:
    for i in imglist:
        # 使用os.path.basename获取文件名
        name = os.path.basename(i)
        f.write('{}\n'.format(name[:-4]))