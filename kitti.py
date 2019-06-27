import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import pptk


DATA_ROOT = '/home/dtc/Data/KITTI/PointPillars'

class_list = ['Car', 'Pedestrian', 'Cyclist']
class_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class Kitti:

    # [0, num_dataset)
    num_dataset = 7481

    def __init__(self, data_root):
        self.training_dataset_dir = os.path.join(data_root, 'training')
        self.testing_dataset_dir = os.path.join(data_root, 'testing')

        print(self.training_dataset_dir)
        print(self.testing_dataset_dir)

    def show_point_cloud(self, pc_id):
        if pc_id < 0 or pc_id >= self.num_dataset:
            sys.exit("pc id is not in [0, " + str(self.num_dataset) + ")")

        # image
        image_filename = os.path.join(self.training_dataset_dir, 'image_2', '{0:06d}.png'.format(pc_id))
        img = mpimg.imread(image_filename)

        # point cloud
        pc_filename = os.path.join(self.training_dataset_dir, 'velodyne_reduced', '{0:06d}.bin'.format(pc_id))
        pc_data = np.fromfile(pc_filename, '<f4')   # little-endian float32
        pc_data = np.reshape(pc_data, (-1, 4))
        pc_color = np.ones((len(pc_data), 3))
        # print(pc_color)
        #
        # v = pptk.viewer(pc_data[:, :3], pc_color)
        # v.set(point_size=0.001)

        # label
        label_filename = os.path.join(self.training_dataset_dir, 'label_2', '{0:06d}.txt'.format(pc_id))

        with open(label_filename) as f_label:
            lines = f_label.readlines()
            for line in lines:
                line = line.strip('\n').split()
                if line[0] in class_list:
                    print(line)
                    box_color = class_color[class_list.index(line[0])]

                    # 2D image
                    left_pixel, top_pixel, right_pixel, bottom_pixel = \
                        int(float(line[4])), int(float(line[5])), int(float(line[6])), int(float(line[7]))
                    label_box_2d = {'left': left_pixel, 'top': top_pixel, 'right': right_pixel, 'bottom': bottom_pixel}
                    self.draw_box_2d(img, label_box_2d, box_color, strode_width=3)

                    # 3D point cloud
                    label_H, label_W, label_L = float(line[8]), float(line[9]), float(line[10])
                    label_X, label_Y, label_Z = float(line[11]), float(line[12]), float(line[13])
                    label_box_3d = {'H': label_H, 'W': label_W, 'L': label_L,
                                    'X': label_X, 'Y': label_Y, 'Z': label_Z}
                    self.draw_box_3d(pc_data, pc_color, label_box_3d, box_color)

        # show 2d
        plt.imshow(img)
        plt.show()

        # show 3d
        v = pptk.viewer(pc_data[:, :3], pc_color)
        v.set(point_size=0.001)


    def draw_box_2d(self, image, box, box_color,  strode_width=1):
        # image is modified
        for i in range(strode_width):
            image[box['top'] + i, box['left']:box['right'], :] = box_color
            image[box['bottom'] - i, box['left']:box['right'], :] = box_color
            image[box['top']:box['bottom'], box['left'] + i, :] = box_color
            image[box['top']:box['bottom'], box['right'] - i, :] = box_color

    def draw_box_3d(self, pc_data, pc_color, box, box_color):
        # pc_color is modified
        for i in range(len(pc_data)):
            # print(pc_data[i])
            if pc_data[i][0] > box['X'] - box['L']/2  and pc_data[i][0] < box['X'] + box['L']/2 \
                and pc_data[i][1] > box['Y'] -box['W']/2 and pc_data[i][1] < box['Y'] + box['W']/2 \
                and pc_data[i][2] > box['Z'] - box['H']/2 and pc_data[i][2] < box['Z'] + box['H']/2:
                pc_color[i][:] = box_color
                print('IN')


if __name__ == '__main__':

    kitti = Kitti(DATA_ROOT)
    kitti.show_point_cloud(0)
