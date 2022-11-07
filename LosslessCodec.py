from copy import deepcopy

import numpy as np
from tqdm import tqdm

from utils import *


# 编码器的建立
class Encoder:
    def __init__(self) -> None:
        pass

    def encode(self, img, path, diffmode=True):
        self.outer = outer(path)
        img = np.array(img, dtype=np.int16)

        # 将图像的尺寸信息写入码流
        bits = tuple()
        h, w = img.shape[:-1]
        bits = bits + uint2bin(h, depth=16) + uint2bin(w, depth=16)
        self.outer.out(bits)

        # 对每一个通道的图像进行差分编码
        if diffmode:
            for c in range(img.shape[2]):
                img[:, :, c] = self._differential_encode(img[:, :, c])
        else:
            pass

        # 对图像进行哈夫曼编码
        self._huffman_encode(img.reshape([-1]))

        self.outer.close()

    # 差分编码
    def _differential_encode(self, img):
        sig = deepcopy(img)
        sig[:, 1:] = img[:, 1:] - img[:, :-1]
        return sig

    # 哈夫曼编码
    def _huffman_encode(self, sig: np.ndarray):
        # 统计图像信息，得到每个像素值的分布概率
        symbs = [i for i in range(sig.min(), sig.max()+1)]
        symbs, probs = hist(sig, symbs)

        # 根据像素值和分布概率建立哈夫曼树
        huffman_dict = huffman(symbs, probs)
        """
        ================ CODE FORMAT ================
        huffman dict:
            +--------+---------------+---------+
            | symbol |  len of code  |   code  |
            +--------+---------------+---------+
            | 9 bits |     5 bits    |  n bits |
            +--------+---------------+---------+
        img codes:
            bits, with 0s filling behind
        =============================================
        """
        # 将哈夫曼表写入码流
        for k in huffman_dict.keys():
            # 使用255移码，将[-255, 255]的像素值放缩到[0, 511]之间
            self.outer.out(uint2bin(k+255, depth=9))
            self.outer.out(uint2bin(len(huffman_dict[k]), depth=5))
            self.outer.out(huffman_dict[k])
        # 以二进制码1 1111 1111作为EOF，分割哈夫曼表区与图像编码区
        self.outer.out(uint2bin(511, depth=9))
        # 将三通道图像展开成向量，进行编码
        sig = np.reshape(sig, [-1])
        # 将每个像素按照哈夫曼编码，写入码流
        for i in tqdm(range(len(sig)), "编码图像"):
            self.outer.out(huffman_dict[sig[i]])

# 解码器的建立


class Decoder:
    def __init__(self) -> None:
        pass

    def decode(self, path: str, diffmode=True):
        self.inner = inner(path)

        # 从码流中获取图像的尺寸信息
        bits = tuple()
        for _ in range(32):
            bits = bits + self.inner.in_()
        h, w = bin2uint(bits[:16]), bin2uint(bits[16:])

        # 对码流进行哈夫曼解码
        img = self._huffman_decode(h, w)

        self.inner.close()

        # 对差分图像进行差分解码
        if diffmode:
            for c in range(img.shape[2]):
                img[:, :, c] = self._differential_decode(img[:, :, c])
        else:
            pass

    # 哈夫曼解码
    def _huffman_decode(self, height: int, width: int) -> np.ndarray:
        """
        ================ CODE FORMAT ================
        huffman dict:
            +--------+---------------+---------+
            | symbol |  len of code  |   code  |
            +--------+---------------+---------+
            | 9 bits |     5 bits    |  n bits |
            +--------+---------------+---------+
        img codes:
            bits, with 0s filling behind
        =============================================
        """
        # 从码流中读取哈夫曼表
        huffman_dict = {}
        while True:
            # 9 bit码流作为像素值，检测到EOF时退出循环，开始读取图像
            symb_bits = tuple()
            for _ in range(9):
                symb_bits = symb_bits + self.inner.in_()
            symb = bin2uint(symb_bits)
            if symb == 511:
                break
            else:
                symb -= 255

            # 5 bit码流作为编码长度值
            len_bits = tuple()
            for _ in range(5):
                len_bits = len_bits + self.inner.in_()
            length = bin2uint(len_bits)

            # 根据读取的编码长度读取编码
            code = tuple()
            for _ in range(length):
                code = code + self.inner.in_()

            # 将键值对写入哈夫曼表
            huffman_dict[code] = symb

        # 开始读取图像
        img = np.zeros([height*width*3], dtype=np.int16)
        codes = huffman_dict.keys()
        bits = tuple()
        for cnt in tqdm(range(height*width*3), "解码图像"):
            while True:
                bits = bits + self.inner.in_()
                # 当文件读取完毕时，self.inner会返回空值，并将self.inner.current_byte设置为-1
                # 用这种方法防止溢出
                if (self.inner.current_byte == -1) or (bits in codes):
                    break
            # 生成图像
            if not self.inner.current_byte == -1:
                img[cnt] = huffman_dict[bits]
            bits = tuple()
        return img.reshape([height, width, 3])

    # 差分解码
    def _differential_decode(self, sig: np.ndarray):
        img = deepcopy(sig)
        for col in range(1, img.shape[1]):
            img[:, col] = sig[:, col] + img[:, col-1]
        return img
