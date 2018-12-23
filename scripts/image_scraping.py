import urllib.request
import cv2
import numpy as np
import os
import pickle


# extracts negative training sets from specific urls
def get_raw_images(neg_images_link):
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    if not os.path.exists('neg'):
        os.makedirs('neg')

    global count

    for _ in neg_image_urls.split('\n'):
        try:
            print(_)
            filename = 'neg/{}.jpg'.format(str(count))
            urllib.request.urlretrieve(_, filename)

            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            resized_img = cv2.resize(img, (100, 100))
            cv2.imwrite(filename, resized_img)

            count += 1

        except Exception as e:
            print(str(e))


# removes broken images from the sample pool
def remove_broken_images(folder):
    for img in os.listdir(folder):
        for broken in os.listdir('broken'):
            try:
                current = folder + '/' + img
                broken_img = cv2.imread('broken/' + broken)
                current_img = cv2.imread(current)

                if img_equals(broken_img, current_img):
                    print(current)
                    os.remove(current)

            except Exception as e:
                print(str(e))


# returns whether two cv2 images are identical
def img_equals(img1, img2):
    return img1.shape == img2.shape and not(np.bitwise_xor(img1, img2).any())


# creates description files for Haar Cascades
def create_des(folder):
    for img in os.listdir(folder):
        if folder == 'neg':
            line = folder + '/' + img + '\n'

        elif folder == 'pos':
            line = folder + '/' + img + ' 1 0 0 50 50\n'

        save_file = open('{}.txt'.format(folder), 'a')
        save_file.write(line)
        save_file.close()


if __name__ == '__main__':
    os.chdir('../data')

    open_file = open('neg_imagenet.pickle', 'rb')
    neg_images_links = pickle.load(open_file)
    open_file.close()

    count = 1

    for link in neg_images_links:
        get_raw_images(link)

    folders = ['neg']

    for folder in folders:
        remove_broken_images(folder)

    for folder in folders:
        create_des(folder)

