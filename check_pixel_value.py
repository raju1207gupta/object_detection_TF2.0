import numpy
import cv2
import matplotlib.pyplot as plt

image_path = 'D:/deeplearning learn/Auto_anotation_tool/sack_dataset/final_datasets/val_seg/SegmentationClass/frame_000000.png'
image = cv2.imread(image_path)
image.shape
plt.imshow(image)
plt.show()

image_gray = cv2.imread(image_path,0)

print ("PIXEL VALUES:\n", "with_sack", image_gray[490, 631], "\nwithout_sack", image_gray[136, 1114])
