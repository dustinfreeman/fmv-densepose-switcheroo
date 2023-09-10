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

def main():
    standard_run()

if __name__ == "__main__":
    main()

