import torch
from inference import image_haze_removal, inference
from PIL import Image
import argparse
import cv2


def process_video(args):
    video_path = args["path"]
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width, frame_height),
    )
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame: {i}")
        img = Image.fromarray(frame)
        dehaze_image = image_haze_removal(img)
        # Convert `dehaze_image` to a numpy array
        dehaze_image = dehaze_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Convert `dehaze_image` to 8-bit unsigned integer
        dehaze_image = (dehaze_image * 255).astype("uint8")
        out.write(dehaze_image)
        i += 1
    cap.release()
    out.release()


if __name__ == "__main__":
    if torch.backends.mps.is_built():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--path",
        required=False,
        help="path to the video",
        default="/Users/flameberry/Developer/Light-DehazeNet/AerialVideo.mp4",
    )
    args = vars(ap.parse_args())
    process_video(args)
