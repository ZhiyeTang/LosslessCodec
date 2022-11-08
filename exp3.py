import os

from PIL import Image

from LosslessCodec import Decoder, Encoder

for i in range(24):
    name = str(i+1).zfill(2)
    print("正在处理[{}/{}]：".format(name, 24))
    img = Image.open("dataset/kodim{}.png".format(name))

    encoder = Encoder()
    encoder.encode(img, "bin/kodim{}.bin".format(name))

    decoder = Decoder()
    decoder.decode("bin/kodim{}.bin".format(name))


result = 0.
for i in range(24):
    name = str(i+1).zfill(2)
    bmp_size = os.path.getsize("./bmp/kodim{}.bmp".format(name))
    bin_size = os.path.getsize("./bin/kodim{}.bin".format(name))
    result += bmp_size / bin_size / 24.
print("最终压缩比：{:.4f}".format(result))
