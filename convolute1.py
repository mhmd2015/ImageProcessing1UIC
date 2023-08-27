import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def convolution_2d(image, filt):
    m, n = filt.shape
    if(m == n):
        y, x = image.shape
        y = y - m +1
        x = x - m +1
        new_image = np.zeros((x,y))
        for i in range(y):
            for j in range(x):
                new_image[i][j]=np.sum(image[i:i+m, j:j+m] * filt)
        return new_image
    

imgPath = 'noise.jpeg'
image = mpimg.imread(imgPath)[::0] #if image RGB just take one channel



filt = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])

result = convolution_2d(image,filt)

fig, axarr = plt.subplots(1,2)

axarr[0].imshow(image,cmap='gray')
axarr[0].set_title('Original Image')


axarr[1].imshow(result,cmap='gray')
axarr[1].set_title('Convolved Image')