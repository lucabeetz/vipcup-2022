from io import BytesIO
from PIL import Image
from random import Random

class ProvidedTransform:
    def __init__(self):
        self.random = Random(42)
        self.output_size = 200
        self.cropsize_min = 160
        self.cropsize_max = 2048
        self.cropsize_ratio = (5, 8)
        self.qf_range = (65, 100)

    def __call__(self, img):
        width, height = img.size

        # Select size of crop
        cropmax = min(min(width, height), self.cropsize_max)
        if cropmax < self.cropsize_min:
            print(width, height)
        assert cropmax >= self.cropsize_min

        cropmin = max(cropmax * self.cropsize_ratio[0] // self.cropsize_ratio[1], self.cropsize_min)
        cropsize = self.random.randint(cropmin, cropmax)

        # Select type of interpolation
        interp = Image.ANTIALIAS if cropsize > self.output_size else Image.CUBIC

        # Select position of crop
        x1 = self.random.randint(0, width - cropsize)
        y1 = self.random.randint(0, height - cropsize) 

        # Select jpeg quality factor
        qf = self.random.randint(*self.qf_range)

        # Make cropping
        img = img.crop((x1, y1, x1+cropsize, y1+cropsize))
        assert img.size[0] == cropsize
        assert img.size[1] == cropsize

        # Make resising
        img = img.resize((self.output_size, self.output_size), interp)
        assert img.size[0] == self.output_size
        assert img.size[1] == self.output_size

        # Make jpeg compression
        outputStream = BytesIO()
        img.save(outputStream, "JPEG", quality = qf)
        outputStream.seek(0)
        img = Image.open(outputStream)

        return img
