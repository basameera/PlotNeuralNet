import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *


def _to_Conv(name, show_h, show_w, show_ch, fil_height, fil_width, fil_depth, offset="(0,0,0)", to="(0,0,0)", name_as_caption=True):
    caption = ''
    if name_as_caption:
        caption = name
    # name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "
    return to_Conv(name, s_filer=show_w, n_filer=show_ch, offset=offset, to=to,
                   height=fil_height//2, depth=fil_width//2, width=fil_depth, caption=caption)


name_as_caption = True
PoseExpNetBN = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_input('image_0_0.png', to='(-1.8,0,0)',
             width=8*2, height=6*2, name="img_n"),
    to_input('image_0_1.png', to='(-0.9,0,0)',
             width=8*2, height=6*2, name="img_n1"),

    _to_Conv('input', 120, 160, 2, 120, 160, 1, offset="(0,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),

    _to_Conv('conv1', 60, 80, 16, 60, 80, 3, offset="(3,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("input", "conv1"),

    _to_Conv('conv2', 30, 40, 32, 30, 40, 6, offset="(5.25,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv1", "conv2"),

    _to_Conv('conv3', 15, 20, 64, 15, 20, 10, offset="(7.5,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv2", "conv3"),

    _to_Conv('conv4', 8, 10, 128, 8, 10, 14, offset="(10,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv3", "conv4"),

    _to_Conv('conv5', 4, 5, 256, 4, 5, 14, offset="(13.4,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv4", "conv5"),

    _to_Conv('conv6', 2, 3, 256, 3, 4, 14, offset="(16.9,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv5", "conv6"),

    _to_Conv('conv7', 1, 2, 256, 2, 3, 14, offset="(20.5,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv6", "conv7"),

    _to_Conv('output', 1, 2, 6, 2, 3, 4, offset="(24,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("conv7", "output"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(PoseExpNetBN, namefile + '.tex')


if __name__ == '__main__':
    main()
