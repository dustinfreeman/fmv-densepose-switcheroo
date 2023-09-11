import argparse
import subprocess
import os

import sys
sys.path.append('../')
sys.path.insert(0, '../DensePoseFnL')
import apply_net

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

def preprocess_video(args):
    # http://trac.ffmpeg.org/wiki/Scaling
    print(f'pre-processing: {args.input_video}')
    input_video_no_ext, ext = os.path.splitext(args.input_video)
    scale_factor = 2

    # proc = subprocess.Popen(['ls', '-la'])
    # test = ' '.join(
    subprocess.call(['ffmpeg', '-i', args.input_video, 
                     '-vf', f'scale=iw/{scale_factor}:ih/{scale_factor}', 
                     input_video_no_ext + '-scaledown' + str(scale_factor) + ext ])
    # print(test)
    # print(proc.args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", type=str,
        help="video for preprocessing")
    args = parser.parse_args()
    preprocess_video(args)

if __name__ == "__main__":
    main()

