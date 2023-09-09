#Shared Python Utilities
import os
import sys
import subprocess

def call(call_obj):
    print(call_obj)
    if isinstance(call_obj, str):
        subprocess.call(call_obj, shell=True)
    else:
        subprocess.call(call_obj)

def prepare_output_folder(output_folder):
    #remove existing pics so we don't have videos accidentally concatenating
    call(['rm', '-rf', output_folder])   # os.path.join(output_folder, '*')])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, mode=777)
    # Need to set chmod again as doing so within os.makedirs does not seem
    # to make this folder accessible to the Docker parent user: 
    call(['chmod', '777', output_folder]) 

assumed_fps = 30    
def vid2imgs(input_video, output_folder, ext='jpg'):
    print('Splitting video: ' + input_video + ' into images in ' + output_folder)

    prepare_output_folder(output_folder)
    
    call(['ffmpeg',\
          '-i', input_video, \
          '-loglevel', 'warning', \
          '-vf', 'fps=' + str(assumed_fps), \
          '-q:v', '1', \
          os.path.join(output_folder, 'frame%06d.' + ext) \
    ])
    #NOTE: -q:v sets quality of jpg to same as video

def imgs2vid(input_folder, output_video, image_file_pattern='frame%06d.jpg'):
    # https://trac.ffmpeg.org/wiki/Slideshow

    call(['ffmpeg', '-framerate', str(assumed_fps), \
          '-y', \
          '-loglevel', 'warning', \
          '-i', os.path.join(input_folder, image_file_pattern), \
          '-pix_fmt', 'yuv420p', \
          #'-c:v', 'libx264', \
          #'-crf', '20', \
          output_video])

def apply_audio(video_recieving_audio, video_providing_audio):
    #add audio from another video (likely the original) to
    # newly generated video
    print('Adding audio from {0} to {1}'.format(video_providing_audio, video_recieving_audio))

    #NOTE: if we don't use a temp video, and instead try to do this in-place,
    # the video writing sequence does not create correct results
    tmp_video = os.path.splitext(video_recieving_audio)[0] + '_tmp.mp4'
    call(['cp', video_recieving_audio, tmp_video])

    call(['ffmpeg', '-i', tmp_video,  \
          '-i', video_providing_audio, \
          '-loglevel', 'warning', \
          '-map', '0:v', '-map', '1:a?', '-c', 'copy', '-shortest', \
          '-y', \
          video_recieving_audio])

    call(['rm', tmp_video])
                    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("utils may be called directly with the name of a subfunction: "+\
              "imgs2vid, vid2imgs, apply_audio")
        exit(1)
    
    subfunction = sys.argv[1]
    #print (subfunction)
    if subfunction == "imgs2vid":
        if len(sys.argv) > 4:
            imgs2vid(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            imgs2vid(sys.argv[2], sys.argv[3])            
    elif subfunction == "vid2imgs":
        if len(sys.argv) > 4:
            vid2imgs(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            vid2imgs(sys.argv[2], sys.argv[3])
    elif subfunction == "apply_audio":
        apply_audio(sys.argv[2], sys.argv[3])
    else:
        print("given subfunction '{0}' not found".format(subfunction))
        exit(1)
        
