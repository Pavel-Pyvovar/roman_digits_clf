from PIL import Image
import numpy as np
import os

def main():
    img_path = os.path.abspath(os.path.join(__file__, '../../../data_clean/1/')) 
    image1 = Image.open(os.path.join(img_path,'1.jpg'))
    image2 = Image.open(os.path.join(img_path,'2.jpg'))
    array1 = np.asarray(image1)
    array2 = np.asarray(image2)
    array = np.stack((array2, ) , axis=0)
    print(array.shape)


if __name__ == "__main__":
    main()