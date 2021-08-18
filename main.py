import numpy as np
import cv2
from PIL import Image
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'C:\\Users\\Lenovo\\PycharmProjects\\pythonProject\\Images\\img.png'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

bw_image = cv2.imread(image_path)

normalized_img = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2LAB)

resized_img = cv2.resize(lab, (224, 244))
L = cv2.split(resized_img)[0]
L -= 50  # subtracting mean value

net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0,:,:,:].transpose((1,2,0))

ab=cv2.resize(ab,(bw_image.shape[1],bw_image.shape[0]))
L=cv2.split(lab)[0]

colorized=np.concatenate((L[:,:,np.newaxis],ab),axis=2)
colorized=cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized=(255.0 * colorized).astype("uint8")

cv2.imshow("BW IMAGE",bw_image)
cv2.imshow("colorized",colorized)

cv2.waitKey(0)
cv2.destroyAllWindows()
def sepia(image_path:str)->Image:
    img = Image.open(image_path)
    width, height = img.size

    pixels = img.load() # create the pixel map

    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr,tg,tb)

    return img