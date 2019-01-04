import cv2
import numpy as np
#import imutils
'''
def addImg(filename, img, all_imgs, x1, y1, x2, y2, class_name):
    if filename not in all_imgs:

        # original
        all_imgs[filename] = {}
        (rows, cols) = img.shape[:2]
        all_imgs[filename]['filepath'] = filename
        all_imgs[filename]['width'] = cols
        all_imgs[filename]['height'] = rows
        all_imgs[filename]['bboxes'] = []
        if np.random.randint(0, 6) > 0:
            all_imgs[filename]['imageset'] = 'trainval'
        else:
            all_imgs[filename]['imageset'] = 'test'

    all_imgs[filename]['bboxes'].append(
        {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)), 'y2': int(float(y2))})


def addImgAndAugmentations(filename, all_imgs, x1, y1, x2, y2, class_name):
    img = cv2.imread(filename)
    (rows, cols) = img.shape[:2]

    # original
    addImg(filename, img, all_imgs, x1, y1, x2, y2, class_name)

    # Flip 90 deg
    flip1Img = imutils.rotate_bound(img, 90)
    addImg(filename+"-flip1", flip1Img, all_imgs, y1, cols-float(x2), y2, cols-float(x1), class_name)

    # Flip 180
    flip2Img = imutils.rotate_bound(img, 180)
    addImg(filename+"-flip2", flip2Img, all_imgs, cols-float(x2), rows-float(y2), cols-float(x1), rows-float(y1), class_name)

    # Flip 270
    flip3Img = imutils.rotate_bound(img, 270)
    addImg(filename+"-flip3", flip3Img, all_imgs, rows-float(y2), x1, rows-float(y1), x2, class_name)

    # High Intensity
    addImg(filename, img + 10, all_imgs, x1, y1, x2, y2, class_name)

    # Low Intensity
    addImg(filename, img - 10, all_imgs, x1, y1, x2, y2, class_name)
'''    


def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
            #addImgAndAugmentations(filename, all_imgs, x1, y1, x2, y2, class_name)
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping

