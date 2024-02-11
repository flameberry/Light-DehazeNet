import torch
import torch.optim
import lightdehazeNet
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu')

def image_haze_removal(input_image):
	hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(hazy_image).float()
	hazy_image = hazy_image.permute(2,0,1)
	hazy_image = hazy_image.to(device).unsqueeze(0)

	ld_net = lightdehazeNet.LightDehaze_Net().to(device)
	ld_net.load_state_dict(torch.load('trained_weights/trained_LDNet.pth'))

	dehaze_image = ld_net(hazy_image)
	return dehaze_image