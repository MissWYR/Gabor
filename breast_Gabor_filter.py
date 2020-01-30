import cv2,os
import numpy as np
import matplotlib.pyplot as plt


def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


#构建Gabor滤波器
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # gabor尺度，6个
     lamda = np.pi/2.0         # 波长
     # for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
     #     for K in range(6):
     #         kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
     #         kern /= 1.5*kern.sum()
     #         filters.append(kern)
     # plt.figure(1)

     for theta in np.arange(0, np.pi, np.pi / 18): #gabor方向，0°，45°，90°，135°，共四个
         kern = cv2.getGaborKernel((5,5), 2, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
         kern /= 1.5*kern.sum()
         filters.append(kern)
     plt.figure(1)

     #用于绘制滤波器
     for temp in range(len(filters)):
         plt.subplot(4, 6, temp + 1)
         plt.imshow(filters[temp])
     plt.show()
     return filters

#Gabor特征提取
def getGabor(img,filters):
    res = [] #滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        id=1
        for kern in filters[i]:
            id+=1
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
            accum_name='Gabor_'+str(id)+'.png'
            #cv2.imwrite(accum_name,accum)
        res.append(np.asarray(accum))

    #用于绘制滤波效果
    plt.figure(2)
    id=1
    for temp in range(len(res)):
        id+=1
        # plt.savefig('Gabor_%d.png'%id)
        temp_name = 'result_' + str(id) + '.png'
        cv2.imwrite(temp_name,res[temp])
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray')
    plt.show()
    return res  #返回滤波结果,结果为24幅图，按照gabor角度排列


if __name__ == '__main__':
    input_Path = './content'
    filters = build_filters()
    img_paths = get_img(input_Path)
    for img in img_paths:
        img = cv2.imread(img)
        getGabor(img, filters)

