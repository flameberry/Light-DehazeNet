from inference import image_haze_removal, inference
from PIL import Image
import torchvision
import os
import argparse


def multiple_dehaze_test(directory):
    print(directory)
    images = []
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename))
        if img is not None:
            images.append(img)

    print(f"Number of Images: {len(images)}")

    c = 0
    for i in range(len(images)):
        img = images[i]
        dehaze_image = image_haze_removal(img)
        torchvision.utils.save_image(
            dehaze_image, "visual_results/dehaze_img(" + str(c + 1) + ").jpg"
        )
        c = c + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-td", "--test_directory", required=False, help="path to test images directory"
    )
    ap.add_argument(
        "-d",
        "--dataset",
        required=False,
        help="path to the parent folder of SS594_Multispectral_Dehazing",
        default="/Users/flameberry/Developer/Dehazing/dataset",
    )
    ap.add_argument(
        "--infer",
        action="store_true",
        help="option to use the test using the SS594_Multispectral_Dehazing dataset",
    )
    args = vars(ap.parse_args())

    if args["infer"]:
        inference(args)
    else:
        multiple_dehaze_test(args["test_directory"])
