import argparse
import subprocess
import os
import sys

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

def preprocess_video(args, skip=False):
    # cheap DAG
    _video = video_downsample(args, skip=True)
    img_split_path = os.path.splitext(args.input_video)[0] + '_imgs'
    if not skip:
        utils.vid2imgs(_video, img_split_path)
    return img_split_path

def apply_densepose_iuv(imgs_path):
    cfg="../DensePoseFnL/configs/mobile_parsing_rcnn_b_s3x.yaml"
    model="/mnt/c/Users/dustinfreeman/Downloads/mobile_parsing_rcnn_b_s3x.pth"

    # imgs_path = "./whimsicals_sandra_behaviour1_imgs/"
    input = imgs_path + "/frame*.jpg"

    apply_net.main(
        [
            "dump", cfg, model, input, 
            "--output", imgs_path + "/iuv_results.pkl", 
            "-v", "--opts", "MODEL.DEVICE", "cpu" 
        ]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=str,
        help="video for preprocessing")
    args = parser.parse_args()
    
    img_split_path = preprocess_video(args, skip=True)

    print(img_split_path)
    apply_densepose_iuv(img_split_path)



if __name__ == "__main__":
    main()

