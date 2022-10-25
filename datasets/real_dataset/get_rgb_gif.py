import os
import cv2

width, height = 1920, 1080

dir = '/mnt/data/hewang/h2o_data/HOI4D/C5/N06/S122/s3/T1/align_image'
lst = []
for i in range(300):
    if i % 2 == 0:
        continue
    image_pth = os.path.join(dir, '%d.jpg' % i)
    lst.append(image_pth)

lst.sort()
img_array = []
for filename in lst:
    img = cv2.imread(filename)
    height_tmp, width_tmp, layers = img.shape
    img = cv2.resize(img, (width//4, height//4))
    size = (width_tmp//4, height_tmp//4)
    img_array.append(img)

out = cv2.VideoWriter('/mnt/data/hewang/h2o_data/video.mp4',0x7634706d, 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
os.system('ffmpeg -i %s -filter:v "setpts=2*PTS" %s' % ('/mnt/data/hewang/h2o_data/video.mp4', '/mnt/data/hewang/h2o_data/slow.mp4'))
os.system('ffmpeg -i %s %s' % ('/mnt/data/hewang/h2o_data/slow.mp4', 'rgb.gif'))