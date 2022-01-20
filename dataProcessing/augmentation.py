import numpy as np
import cv2
import random
import albumentations as A

imgs = []
boxes = []


def read_images(max_id):
    for i in range(1, max_id+1):
        img_url = 'F:/20211/DeepLearning/Test/dataset/images/train/' + str(i) + '.jpg'
        label_url = 'F:/20211/DeepLearning/Test/dataset/labels/train/' + str(i) + '.txt'
        img = cv2.imread(img_url)
        if img is not None:
            img = cv2.resize(img, (1280, 720))
            imgs.append(img)
            with open(label_url) as f:
                this_boxes = []
                for label in f.readlines():
                    class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n","").split()]
                    this_boxes.append([x, y, width, height, class_label])
                boxes.append(this_boxes)


def string_format(bboxes):
    res = ""
    for bbox in bboxes:
        x, y, width, height, class_label = bbox
        res += str(class_label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n'
    return res


def save_img(img, box, index):
    img_url = 'F:/20211/DeepLearning/Test/dataset/images/train/' + str(index) + '.jpg'
    label_url = 'F:/20211/DeepLearning/Test/dataset/labels/train/' + str(index) + '.txt'
    cv2.imwrite(img_url, img)
    f = open(label_url, 'w')
    f.write(string_format(box))
    f.close()


# cut out
def cutout(image):
    for i in range(random.randint(15, 20)):
        x = random.randint(0, 1260)
        y = random.randint(0, 700)
        image[y:(y + 20), x:(x + 20)] = [0 for _ in range(3)]
    return image


transform = A.Compose([
    A.Resize(width=1280,height=720),
    A.Rotate(limit=10,p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.RGBShift(r_shift_limit=25,g_shift_limit=25,b_shift_limit=25,p=0.9)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.6))


# mosaic
def create_mosaic(amount, start_index):
    mosaic_images = []
    mosaic_boxes = []
    for i in range(amount):
        img4_id_list = random.sample(range(len(imgs)), 4)
        big_img = np.full((1440, 2560, imgs[0].shape[2]), 0, dtype=np.uint8)
        this_boxes = []
        for j in range(len(img4_id_list)):
            if j == 0:
                big_img[:720, :1280] = imgs[img4_id_list[j]]
                this_boxes.extend([box[0]/2, box[1]/2, box[2]/2, box[3]/2, box[4]] for box in boxes[img4_id_list[j]])
            if j == 1:
                big_img[:720, 1280:] = imgs[img4_id_list[j]]
                this_boxes.extend([box[0]/2+0.5, box[1]/2, box[2] / 2, box[3] / 2, box[4]] for box in boxes[img4_id_list[j]])
            if j == 2:
                big_img[720:, :1280] = imgs[img4_id_list[j]]
                this_boxes.extend([box[0]/2, box[1]/2+0.5, box[2] / 2, box[3] / 2, box[4]] for box in boxes[img4_id_list[j]])
            if j == 3:
                big_img[720:, 1280:] = imgs[img4_id_list[j]]
                this_boxes.extend([box[0]/2+0.5, box[1]/2+0.5, box[2] / 2, box[3] / 2, box[4]] for box in boxes[img4_id_list[j]])
        mosaic_img = cv2.resize(big_img, (1280, 720))
        mosaic_img = cutout(mosaic_img)

        transformed = transform(image=mosaic_img, bboxes=this_boxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        mosaic_images.append(transformed_image)
        mosaic_boxes.append(transformed_bboxes)
        save_img(transformed_image, transformed_bboxes, start_index + i)
    return mosaic_images, mosaic_boxes



def create_mix_up(amount, start_index):
    mix_up_imgs = []
    mix_up_boxes = []
    for i in range(amount):
        img2_id_list = random.sample(range(len(imgs)), 2)
        this_boxes = boxes[img2_id_list[0]]
        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        mix_up_img = (imgs[img2_id_list[0]] * r + imgs[img2_id_list[1]] * (1 - r)).astype(np.uint8)
        this_boxes.extend(boxes[img2_id_list[1]])

        mix_up_img = cutout(mix_up_img)
        transformed = transform(image=mix_up_img, bboxes=this_boxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        save_img(transformed_image, transformed_bboxes, start_index + i)

        mix_up_imgs.append(transformed_image)
        mix_up_boxes.append(transformed_bboxes)
    return mix_up_imgs, mix_up_boxes


read_images(1015)
create_mosaic(1000, 1016)
create_mix_up(1000, 2016)
