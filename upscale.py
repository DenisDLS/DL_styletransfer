import cv2
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=2, padding_mode='replicate')
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0


def padding(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]
    return zimg


def bicubic(img, ratio, a):
    H, W, C = img.shape
    img = padding(img, H, W, C)
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, 3))
    h = 1 / ratio
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2
                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x
                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y
                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c],
                                    img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c],
                                    img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c],
                                    img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c],
                                    img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c],
                                    img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c],
                                    img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c],
                                    img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c],
                                    img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix(
                    [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst


def result_photo(src_0, scale=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('srcnn.pth'))
    outp = []
    image = cv2.imread(src_0, cv2.IMREAD_COLOR)
    if scale != 1:
        image = bicubic(image, scale, -1 / 2)
    for i in range(3):
        im = image[:, :, i]
        im = im.reshape(im.shape[0], im.shape[1], 1)
        im = im / 255.
        model.eval()
        with torch.no_grad():
            im = np.transpose(im, (2, 0, 1)).astype(np.float32)
            im = torch.tensor(im, dtype=torch.float).to(device)
            im = im.unsqueeze(0)
            outputs = model(im)
            outputs = outputs.cpu()
            outputs = outputs.detach().numpy()
            outputs = outputs.reshape(outputs.shape[2], outputs.shape[3], outputs.shape[1])
            outp.append(outputs)
    res = np.concatenate((outp[0], outp[1], outp[2]), axis=2)
    src = "tmp/result.jpg"
    cv2.imwrite(src, res * 255)