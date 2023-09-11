import argparse
import subprocess
import os
import sys
from PIL import Image
import pickle
from pathlib import Path

import numpy as np

sys.path.append('../')
sys.path.insert(0, '../DensePoseFnL')
import apply_net
sys.path.insert(0, '../UVTextureConverter')
from UVTextureConverter import UVConverter

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
    input_video_no_ext, ext = os.path.splitext(args.preprocess)
    
    scale_factor = 4
    output_filename = input_video_no_ext + '-scaledown' + str(scale_factor) + ext

    if not skip:
        # proc = subprocess.Popen(['ls', '-la'])
        # test = ' '.join(
        subprocess.call(['ffmpeg', '-i', args.preprocess, 
                        '-vf', f'scale=iw/{scale_factor}:ih/{scale_factor}', 
                        input_video_no_ext + '-scaledown' + str(scale_factor) + ext ])
        # print(test)
        # print(proc.args)
    return output_filename

def _processing_base_path(args):
    return os.path.splitext(args.preprocess)[0] + '/'

def _img_split_path(args):
    img_split_path = _processing_base_path(args) + 'frames'
    return img_split_path

def preprocess_video(args, skip=False):
    # cheap DAG
    _video = video_downsample(args, skip=False)
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

        bg_img.save(_iuv_images_path(args) + file_shortname_noext + '.png') 

def _static_texture_path(args):
    return _processing_base_path(args) + "static_texture.png"

parts_size = 16
def create_static_texture(args):
    im_list = list(Path(_img_split_path(args)).iterdir())
    im_list = [str(im) for im in im_list] 

    iuv_list = list(Path(_iuv_images_path(args)).iterdir())
    iuv_list = [str(im) for im in iuv_list] 

    tex_im, mask_im = UVConverter.create_texture_from_video(im_list, iuv_list, parts_size=parts_size)
    static_text_im = Image.fromarray(np.uint8(tex_im * 255),"RGB")
    static_text_im.save(_static_texture_path(args))

# movie.mov
# movie_scaledown2.mov
# movie/
# movie/frames/...*.jpg
# movie/iuv_results.pkl
# movie/iuv/...*.png
# movieA-to-movieB/frames/*.jpg
# movieA-to-movieB.mov

def _transfer_result_path(args):
    return os.path.splitext(args.source)[0] + '-to-' + \
        os.path.splitext(args.dest)[0].replace('../', '') +'/'

def transfer_texture(args):
    # transfer texture from source static texture to dest images
    static_texture = os.path.splitext(args.source)[0] + '/' +  'static_texture.png'
    dest_iuv_list = list(Path(os.path.splitext(args.dest)[0] + '/iuv').iterdir())
    dest_iuv_list = [str(im) for im in dest_iuv_list] 

    utils.prepare_output_folder(_transfer_result_path(args) + 'frames/')

    i_id, u_id, v_id = 2, 1, 0
    parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    
    static_texture = Image.open(static_texture)
    static_texture = (np.array(static_texture)).transpose(0, 1, 2)

    for iuv_file in dest_iuv_list:
        dest_iuv = Image.open(iuv_file)
        iuv = (np.array(dest_iuv)).transpose(2, 1, 0)

        image_out = np.zeros((3, iuv.shape[1], iuv.shape[2]))
        for x in range(image_out.shape[1]):
            for y in range(image_out.shape[2]):
                # determine part_id 
                part_id = iuv[i_id][x][y]
                if part_id == 0:
                    # no body, ignore
                    continue
                if part_id > parts_list[-1]:
                    # print(f'!! out of bounds part_id {part_id} {(x, y)}')
                    continue
                # sample from correct part sub_image in iuv
                u = iuv[u_id][x][y]
                v = iuv[v_id][x][y]
                atlas_subimage = ( (part_id - 1) % 6, (part_id - 1) // 6)
                uv_coords = (u / 255, v / 255)

                tex_coords = (
                    atlas_subimage[0] * parts_size + 
                        int((u * (parts_size - 1)) / 255),
                    atlas_subimage[1] * parts_size + 
                        int((v * (parts_size - 1)) / 255)
                )

                for c in [0, 1, 2]:
                    # print(c, x, y, tex_coords)
                    image_out[c, x, y] = static_texture\
                                            [tex_coords[1]][tex_coords[0]][c]
        
        _im_out = Image.fromarray(np.uint8(image_out.transpose(2, 1, 0)), "RGB")

        iuv_file_shortname = iuv_file.split('/')[-1]
        # iuv_file_shortname_noext = os.path.splitext(iuv_file_shortname)[0]
        _im_out.save(_transfer_result_path(args) + 'frames/' + iuv_file_shortname)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-preprocess", type=str,
        help="video for preprocessing")
    parser.add_argument("-source", type=str,
        help="take texture from this video")
    parser.add_argument("-dest", type=str,
        help="apply texture to this video")
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_video(args, skip=False)
        apply_densepose_iuv(args)
        create_iuv_images(args)
        create_static_texture(args)
    if args.source and args.dest:
        transfer_texture(args)
        utils.imgs2vid(_transfer_result_path(args) + 'frames/', _transfer_result_path(args) + 'result.mp4')

if __name__ == "__main__":
    main()

