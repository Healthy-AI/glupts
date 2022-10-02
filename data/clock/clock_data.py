from data.linear_gaussian import linear_gaussian_system
import numpy as np
import math
from PIL import Image, ImageDraw


MAX_SINGULAR_VAL = 10

class clock_image_visualizer():
    def __init__(self):
        self.target_size = 28
        self.size = 200
        self.scaling = self.size / self.target_size
        self.width = self.size
        self.height = self.size
        self.center = ((self.width + 1) / 2, (self.width + 1) / 2)
        self.inner_arm_length = 8
        self.outer_arm_length = 10
        self.white = 255
        self.black = 0
        self.full_rotation_threshold = 3
        self.added_rotation = math.pi / 2
        self.inner_circle_size = 2

    def get_limit(self):
        return (self.full_rotation_threshold + 0.99) * 2 * math.pi

    def xy_to_canvas(self, x, y):
        x = self.scaling * x + self.center[0]
        y = self.scaling * y + self.center[1]
        return (x, y)

    def draw_circle(self, drawObj, center, radius, **kwargs):
        radius = radius * self.scaling
        corner_dist = math.sqrt(2 * radius ** 2) / 2
        left_upper = (center[0] - corner_dist, center[1] - corner_dist)
        right_lower = (center[0] + corner_dist, center[1] + corner_dist)
        drawObj.ellipse((left_upper, right_lower), **kwargs)

    def calculate_properties(self, x, armlength=1):
        positive = int(x >= 0.0)
        full_rotations = abs(int(x / (2 * math.pi)))
        partial_rotation = x % (2 * math.pi)

        if full_rotations > self.full_rotation_threshold:
            raise Exception(f'More than {self.full_rotation_threshold} spins are hard to visualize! '
                            f'You asked for {full_rotations} rotations. Input number: {x}')
        pos_x = math.cos(partial_rotation + self.added_rotation) * armlength
        pos_y = -math.sin(partial_rotation + self.added_rotation) * armlength

        canvas_coords = self.xy_to_canvas(pos_x, pos_y)
        return positive, full_rotations, canvas_coords

    def draw_inner_arm(self, d, x):
        positive, full_rotations, coords = self.calculate_properties(x, self.inner_arm_length)
        d.line([self.center, coords], fill=self.white, width=round(2 * self.scaling))
        inner_circle_size = int(full_rotations + 2)
        self.draw_circle(d, self.center, inner_circle_size + 1, fill=self.white)
        self.draw_circle(d, self.center, inner_circle_size, fill=positive * self.white)

    def draw_outer_arm(self, d, x):
        positive, full_rotations, coords = self.calculate_properties(x, self.outer_arm_length)
        self.draw_circle(d, coords, (full_rotations + 2) + 1, fill=self.white)
        self.draw_circle(d, coords, (full_rotations + 2), fill=positive * self.white)

    def paint_image_as_array(self, num1, num2):
        num1 = float(num1)
        num2 = float(num2)
        image = Image.new('L', (self.width, self.height), color=self.black)
        d = ImageDraw.Draw(image)

        self.draw_inner_arm(d, num1)
        self.draw_outer_arm(d, num2)
        image = image.resize((self.target_size, self.target_size), resample=Image.Resampling.LANCZOS)
        return np.array(image) / self.white

    def visualize_image_array(self, array):
        im = self.array2image(array)
        im.show()
        return im

    def array2image(self, array):
        return Image.fromarray((array * self.white).astype(np.uint8), mode='L')

    def visualize_sequence(self, seq):
        assert len(seq.shape) == 3, f'Expected 3 dimensions, got {len(seq.shape)}'
        n, h, w = seq.shape
        array = seq.reshape(h * n, w)
        return self.visualize_image_array(array)


class clock_LGS(linear_gaussian_system):
    def __init__(self, y_features=1, test_size=1000, transition_noise=1, seed=0):
        assert y_features in [1,2], 'Can only support 1 or 2 features with the clock data set.'
        super(clock_LGS, self).__init__(2, y_features, test_size, seed)
        self.name = f'Clock LGS Y{y_features} N{transition_noise}'
        self.y_features = y_features
        self.trans_noise = transition_noise
        self.test_size = test_size
        self.painter = clock_image_visualizer()

    def get_train_test_data(self, seq_length, seq_step, test_ratio, sample_size, seed, copy_instead_of_split=False):
        enough_data = False
        current_sampling_size_tr = sample_size
        original_test_size = self.test_size
        while not enough_data:
            current_sampling_size_tr *= 2
            self.test_size *= 2
            (z_tr, y_tr), (z_te, y_te) = super(clock_LGS, self).get_train_test_data(seq_length, seq_step,
                                                                                    test_ratio, current_sampling_size_tr,
                                                                                    seed,
                                                                                    copy_instead_of_split,
                                                                                    noise_std=self.trans_noise)
            z_tr, y_tr = self.filter_by_painter_limit(z_tr, y_tr)
            z_te, y_te = self.filter_by_painter_limit(z_te, y_te)
            enough_data = z_tr.shape[0] >= sample_size and z_te.shape[0] >= original_test_size
        self.test_size = original_test_size

        z_tr = z_tr[:sample_size]
        z_te = z_te[:self.test_size]
        y_tr = y_tr[:sample_size]
        y_te = y_te[:self.test_size]


        x_tr = self.transform_z_to_x(z_tr)
        x_te = self.transform_z_to_x(z_te)

        N, T, W, H = x_tr.shape
        x_tr = x_tr.reshape(N,T, W * H)
        N, T, W, H = x_te.shape
        x_te = x_te.reshape(N, T, W * H)

        self.latent_variables = (z_tr, z_te)

        return (x_tr, y_tr), (x_te, y_te)

    def get_info(self):
        return {'Output_Features': self.y_features, 'Trans_Noise': self.trans_noise, 'Test Size': self.test_size,
                **super(clock_LGS, self).get_info()}

    def filter_by_painter_limit(self, z,y):
        limit = self.painter.get_limit()
        idx = ~np.max(np.abs(z) > limit,(1,2))
        return z[idx], y[idx]

    def get_latent_vars(self):
        return self.latent_variables

    def transform_z_to_x(self, z):
        N,T,D = z.shape
        assert D==2
        samples = []
        for i in range(N):
            sequence = []
            for t in range(T):
                sequence.append(self.painter.paint_image_as_array(z[i,t,0], z[i,t,1]))
            samples.append(np.stack(sequence,0))
        return np.stack(samples,0)

if __name__ == '__main__':

    dataset = clock_LGS(2)
    (train_x, train_y), (test_x, test_y) = dataset.get_train_test_data(6, 1, 0.2, 2, 81)
    dataset.painter.visualize_image_array(train_x[0].reshape(-1,28,28)[-1])
    print(dataset.get_latent_vars()[0][0,-1])
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # for _ in range(1):
    #     dataset.painter.visual_sequence(train_x[int(np.random.randint(0,200,1))])
    #dataset.painter.visualize_image_array(test_x[0,0].reshape(28,28))
    _, T, _ = test_x.shape

    # images = []
    # for k in range(5):
    #     im = test_x.reshape(-1, T, 28, 28)[k]
    #     new_im = np.zeros((im.shape[0], 30,30)) +0.2
    #     new_im[:,1:-1,1:-1] = im
    #     n, h, w = new_im.shape
    #     array = new_im.reshape(h * n, w)
    #     images.append(array)
    #     print(array.shape)
    # tableau = np.concatenate(images,1)
    # dataset.painter.visualize_image_array(tableau)
    # print(dataset.coefficients)
    # dataset = clock_LGS(2)
    # (train_x, train_y), (test_x, test_y) = dataset.get_train_test_data(6, 1, 0.2, 200, 0)
    # print(dataset.coefficients)