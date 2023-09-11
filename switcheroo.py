import argparse
import subprocess
import os
import sys
from PIL import Image
import pickle

import numpy as np

sys.path.append('../')
sys.path.insert(0, '../DensePoseFnL')
import apply_net

import utils


def standard_run():
    # python ./DensePoseFnL/apply_net.py show --output /mnt/c/Users/dustinfreeman/Downloads/wsl-output.jpg ./DensePoseFnL/configs/mobile_parsing_rcnn_b_s3x.yaml /mnt/c/Users/dustinfreeman/Downloads/mobile_parsing_rcnn_b_s3x.pth /mnt/c/Users/dustinfreeman/Dropbox/Research/Sabbatical/FMV\ Game\ Jam/fmv-jam-cc4-assets/north_small/starter_dustin_static.jpg dp_contour -v --opts MODEL.DEVICE cpu 
    cfg="../DensePoseFnL/configs/mobile_parsing_rcnn_b_s3x.yaml"
    model="/mnt/c/Users/dustinfreeman/Downloads/mobile_parsing_rcnn_b_s3x.pth"
    input="/mnt/c/Users/dustinfreeman/Dropbox/Research/Sabbatical/FMV Game Jam/fmv-jam-cc4-assets/north_small/starter_dustin_static.jpg"
    apply_net.main(
        [
            "show", cfg, model, input, 
            "--output", "/mnt/c/Users/dustinfreeman/Downloads/wsl-output.jpg", 
            "dp_contour", "-v", "--opts", "MODEL.DEVICE", "cpu" 
        ]
    )
    # show = apply_net.ShowAction()
    # show.execute(
    #     {"cfg": cfg, "model": model, "input": input,
    #     "opts":[["MODEL.DEVICE", "cpu"]],
    #     "visualizers":["dp_contour"],
    #     "output":"/mnt/c/Users/dustinfreeman/Downloads/wsl-output.jpg"
    # })

def video_downsample(args, skip=False):
    # http://trac.ffmpeg.org/wiki/Scaling
    input_video_no_ext, ext = os.path.splitext(args.input_video)
    
    scale_factor = 2
    output_filename = input_video_no_ext + '-scaledown' + str(scale_factor) + ext

    if not skip:
        # proc = subprocess.Popen(['ls', '-la'])
        # test = ' '.join(
        subprocess.call(['ffmpeg', '-i', args.input_video, 
                        '-vf', f'scale=iw/{scale_factor}:ih/{scale_factor}', 
                        input_video_no_ext + '-scaledown' + str(scale_factor) + ext ])
        # print(test)
        # print(proc.args)
    return output_filename

def _processing_base_path(args):
    return os.path.splitext(args.input_video)[0] + '/'

def _img_split_path(args):
    img_split_path = _processing_base_path(args) + 'frames'
    return img_split_path

def preprocess_video(args, skip=False):
    # cheap DAG
    _video = video_downsample(args, skip=True)
    if not skip:
        utils.vid2imgs(_video, _img_split_path(args))

def _iuv_results_path(args):
    return _processing_base_path(args) + "iuv_results.pkl"

def apply_densepose_iuv(args):
    cfg="../DensePoseFnL/configs/mobile_parsing_rcnn_b_s3x.yaml"
    model="/mnt/c/Users/dustinfreeman/Downloads/mobile_parsing_rcnn_b_s3x.pth"

    # imgs_path = "./whimsicals_sandra_behaviour1_imgs/"
    input = _img_split_path(args) + "/frame*.jpg"

    apply_net.go(
        [
            "dump", cfg, model, input, 
            "--output", _iuv_results_path(args), 
            "-v", "--opts", "MODEL.DEVICE", "cpu" 
        ]
    )

def _iuv_images_path(args):
    return _processing_base_path(args) + "iuv/"

def create_iuv_images(args):
    # get image dimensions
    img = Image.open(_img_split_path(args) + "/frame000001.jpg")
    img_w, img_h = img.size
    # img_w, img_h

    with open(_iuv_results_path(args),'rb') as f:
        data=pickle.load(f)

    utils.prepare_output_folder(_iuv_images_path(args))

    for img_file in data:
        file_shortname = img_file['file_name'].split('/')[-1]
        file_shortname_noext = os.path.splitext(file_shortname)[0]

        i = img_file['pred_densepose'][0].labels.cpu().numpy()
        uv = img_file['pred_densepose'][0].uv.cpu().numpy()
        
        iuv = np.stack((
                uv[1,:,:]*255, 
                uv[0,:,:]*255, 
                i
            ))
        iuv = np.transpose(iuv, (1,2,0))
        iuv_img = Image.fromarray(np.uint8(iuv),"RGB")

        #uncrop to entire original image
        box = img_file["pred_boxes_XYXY"][0]
        box[2]=box[2]-box[0]
        box[3]=box[3]-box[1]
        x,y,w,h=[int(v) for v in box]
        bg=np.zeros((img_h,img_w,3))
        bg[y:y+h,x:x+w,:]=iuv
        bg_img = Image.fromarray(np.uint8(bg),"RGB")

        bg_img.save(_iuv_images_path(args) + file_shortname_noext + '_iuv.png') 

# movie.mov
# movie_scaledown2.mov
# movie/
# movie/frames/...*.jpg
# movie/iuv_results.pkl
# movie/iuv/...*.png

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=str,
        help="video for preprocessing")
    args = parser.parse_args()
    
    # preprocess_video(args, skip=True)
    # apply_densepose_iuv(args)
    create_iuv_images(args)


if __name__ == "__main__":
    main()

