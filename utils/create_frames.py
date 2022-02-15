import cv2
import time
import os

def video_to_frames(input_loc, output_loc, interval = 10):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.makedirs(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    count_ = 0
    print ("Converting video..\n")
    print("saving to ", output_loc)
    print(os.path.isdir(output_loc))
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        if count % interval == 0:
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count_+1), frame)
            count_ += 1
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__":

    input_loc = '../custom_data/'
    output_loc = '../custom_data/frames/'
    datasets = ['1','2','8','9']

    for i in datasets:
        vid_file = os.path.join(input_loc, i + '.mp4')
        print(vid_file, os.path.isfile(vid_file))
        output_dir = os.path.join(output_loc, i)
        print(output_dir)
        if i == '1':
            video_to_frames(vid_file, output_dir, interval = 25)
        else:
            video_to_frames(vid_file, output_dir)

    # print(os.path.isdir(input_loc))
    # video_to_frames(input_loc, output_loc)