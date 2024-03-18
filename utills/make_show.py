import glob
import PIL
import base64
from IPython import display

def make_gif(set_dir_pil,
             out_name = "my_awesome.gif",
             duration = 10,
             loop=0,
             ext = '/*.JPG'):

    if type(set_dir_pil) == str:
       frames = [PIL.Image.open(image) for image in glob.glob(f"{set_dir_pil}{ext}")]

    elif type(set_dir_pil) == list: frames = set_dir_pil

    frame_one = frames[0]
    frame_one.save(out_name, format="GIF", append_images=frames,
               save_all=True, optimize=False, duration=duration, loop=loop)


def show_gif(fname):
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')