import cv2
import numpy as np

class Dataset(object):
    def __init__(self, uid, df, crop_size, resize_size):
        index = np.where(df['uid_date'].isin(uid))[0]
        images = []
        labels = []
        weights = []
        for i in index:
            img = self._load_image(df['path'][i], crop_size, resize_size)
            label = df['egfr_mdrd'][i]
            if label < 30:
                weight = 3
            elif label < 60:
                weight = 1
            elif label < 90:
                weight = 3
            else:
                weight = 9
            images.append(img)
            labels.append(label)
            weights.append(weight)
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.weights = np.array(weights)
    
    def _load_image(self, path, crop_size, resize_size):
        # load image
        img = cv2.imread(path,1)
        # we crop image from center
        yy = int((img.shape[0] - crop_size) / 2)
        xx = int((img.shape[1] - crop_size) / 2)
        crop_img = img[yy: yy + crop_size, xx: xx + crop_size]
        # resize to 224, 224
        resized_img = cv2.resize(crop_img,(resize_size,resize_size), interpolation=cv2.INTER_AREA)
        return resized_img
    
    def shuffle(self):
        idx = np.random.permutation(self.images.shape[0])
        self.images = self.images[idx,:,:,:]
        self.labels = self.labels[idx]
        self.weights = self.weights[idx]