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

    _to_Conv('fmap1', 60, 80, 16, 100, 140, 3, offset="(2,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("input", "fmap1"),

    _to_Conv('fmap2', 30, 40, 32, 80, 120, 5, offset="(4.2,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap1", "fmap2"),

    _to_Conv('fmap3', 15, 20, 64, 60, 100, 9, offset="(6.5,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap2", "fmap3"),

    _to_Conv('fmap4', 8, 10, 128, 40, 80, 12, offset="(9.5,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap3", "fmap4"),

    _to_Conv('fmap5', 4, 5, 256, 20, 60, 12, offset="(12.5,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap4", "fmap5"),

    _to_Conv('fmap6', 2, 3, 256, 10, 40, 12, offset="(16,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap5", "fmap6"),

    _to_Conv('fmap7', 1, 2, 256, 7, 30, 12, offset="(19.2,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap6", "fmap7"),

    _to_Conv('fmap8', 1, 2, 6, 4, 4, 4, offset="(22.3,0,0)",
             to="(0,0,0)", name_as_caption=name_as_caption),
    to_connection("fmap7", "fmap8"),

    to_SoftMax("fc1", 512 ,"(1.5,0,0)", "(fmap8-east)", width=1.5, height=3, depth=50, caption="fmap9"),
    to_connection("fmap8", "fc1"),   

    to_SoftMax("fc2", 256 ,"(1.5,0,0)", "(fc1-east)", width=1.5, height=3, depth=25, caption="fmap10"),
    to_connection("fc1", "fc2"),

    to_SoftMax("output", 6 ,"(1.5,0,0)", "(fc2-east)", width=1.5, height=3, depth=15, caption="output"),
    to_connection("fc2", "output"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(PoseExpNetBN, namefile + '.tex')


if __name__ == '__main__':
    main()
