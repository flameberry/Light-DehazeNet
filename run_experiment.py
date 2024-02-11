import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import lightdehazeNet
from torchvision import transforms
import pathlib

from DehazeDataset import DehazingDataset, DatasetType
from PIL import Image
import cv2

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_built() else "cpu"
)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def Preprocess(image: Image.Image) -> torch.Tensor:
    # Contrast Enhancement
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
            ),
            # transforms.functional.equalize
        ]
    )
    transformedImage = transform(image)

    # Gamma Correction
    gammaCorrectedImage = transforms.functional.adjust_gamma(transformedImage, 2.2)

    # Histogram Stretching
    min_val = gammaCorrectedImage.min()
    max_val = gammaCorrectedImage.max()
    stretchedImage = (gammaCorrectedImage - min_val) / (max_val - min_val)

    # Guided Filtering
    gFilter = cv2.ximgproc.createGuidedFilter(
        guide=stretchedImage.permute(1, 2, 0).numpy(), radius=3, eps=0.01
    )
    filteredImage = gFilter.filter(src=stretchedImage.permute(1, 2, 0).numpy())
    return torch.from_numpy(filteredImage).permute(2, 0, 1)


def train(args):
    ld_net = lightdehazeNet.LightDehaze_Net().to(device)
    ld_net.apply(weights_init)

    parentPath = pathlib.Path(args["dataset"])
    dehazingDatasetPath = parentPath / "SS594_Multispectral_Dehazing/Haze1k/Haze1k"

    training_data = DehazingDataset(
        dehazingDatasetPath=dehazingDatasetPath,
        _type=DatasetType.Train,
        transformFn=Preprocess,
        verbose=False,
    )
    validation_data = DehazingDataset(
        dehazingDatasetPath=dehazingDatasetPath,
        _type=DatasetType.Validation,
        transformFn=Preprocess,
        verbose=False,
    )

    training_data_loader = torch.utils.data.DataLoader(
        training_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )
    validation_data_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        ld_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001
    )

    print("Started Training...")

    losses = []

    num_of_epochs = int(args["epochs"])
    for epoch in range(num_of_epochs):
        print(f"Epoch {epoch + 1}/{num_of_epochs}")

        ld_net.train()
        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):
            hazefree_image = hazefree_image.to(device)
            hazy_image = hazy_image.to(device)

            dehaze_image = ld_net(hazy_image)

            loss = criterion(dehaze_image, hazefree_image)

            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ld_net.parameters(), 0.1)
            optimizer.step()

            if ((iteration + 1) % 10) == 0:
                print(f"\tStep {iteration + 1}: Loss - {loss.item()}")

            if ((iteration + 1) % 200) == 0:
                print("Saving Model...")
                torch.save(
                    ld_net.state_dict(),
                    "trained_weights/" + "Epoch" + str(epoch) + ".pth",
                )

        # Validation Stage
        print("Saving Validation Images...")

        print("SSIM\tSSIM_Previous\tPSNR\tPSNR_Previous")

        ld_net.eval()
        for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):
            hazefree_image = hazefree_image.to(device)
            hazy_image = hazy_image.to(device)

            dehaze_image = ld_net(hazy_image)

            # Calculate and print the SSIM
            ssim = StructuralSimilarityIndexMeasure().to(device)
            ssim_val = ssim(dehaze_image, hazefree_image)
            ssim_fake_val = ssim(hazy_image, hazefree_image)

            # Calculate and print the PSNR
            psnr = PeakSignalNoiseRatio().to(device)
            psnr_val = psnr(dehaze_image, hazefree_image)
            psnr_fake_val = psnr(hazy_image, hazefree_image)

            print(
                f"{ssim_val:.4f}\t|\t{ssim_fake_val:.4f}\t|\t{psnr_val:.4f}\t|\t{psnr_fake_val:.4f}"
            )

            save_path = (
                "training_data_captures/"
                + f"Epoch_{epoch}_Step_{str(iter_val+1)}"
                + ".jpg"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(
                torch.cat((hazy_image, dehaze_image, hazefree_image), 0), save_path
            )

        torch.save(ld_net.state_dict(), "trained_weights/" + "trained_LDNet.pth")

    # Store the graph of the loss
    plt.plot(losses)
    plt.savefig("LossPlot.png")
    # plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        required=False,
        help="path to the parent folder of SS594_Multispectral_Dehazing",
        default="/Users/flameberry/Developer/Dehazing/dataset",
    )
    ap.add_argument(
        "-e", "--epochs", required=True, help="number of epochs for training"
    )
    ap.add_argument(
        "-lr", "--learning_rate", required=True, help="learning rate for training"
    )

    args = vars(ap.parse_args())

    train(args)
