import cv2
import numpy as np
import os
import model
import torch
from skimage import io
import torchvision.transforms as transforms
labels={'0':'anudeep','1':'saikrishna','2':'sricharan','3':'varun','4':'unknown'}
test_img=cv2.imread('test.jpg')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
gray= cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=5)
transform = transforms.ToTensor()

vgg=model.vgg()
vgg.load_state_dict(torch.load('./params/params2.pth'))
for (x, y, w, h) in faces:
    img=test_img[y:y+h, x:x+w]
    image = np.array(img, "uint8")
    image=transform(image)
    reshaped =image.permute(0,1,2).unsqueeze(0)
    print('processing........')
    out = vgg(reshaped)
    print('generating output....\nPlease wait...')
    _, pred = out.max(1)
    txt = labels[str(int(pred[0]))]
    # test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)
    cv2.putText(test_img, txt, (x, y), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 0), 6)
print('image generated')
test_img=cv2.resize(test_img,(550,550))
cv2.imshow('image',test_img)
if cv2.waitKey() & 0xFF == ord('q'):
    pass
cv2.destroyAllWindows()