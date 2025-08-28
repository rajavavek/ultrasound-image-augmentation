import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import collections
from src.config import cfg

class EqualizedLR_Conv2d(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
		super().__init__()
		self.padding = padding
		self.stride = stride
		self.scale = np.sqrt(2/(in_ch * kernel_size[0] * kernel_size[1]))

		self.weight = Parameter(torch.Tensor(out_ch, in_ch, *kernel_size))
		self.bias = Parameter(torch.Tensor(out_ch))

		nn.init.normal_(self.weight)
		nn.init.zeros_(self.bias)

	def forward(self, x):
		return F.conv2d(x, self.weight*self.scale, self.bias, self.stride, self.padding)
	
class Minibatch_std(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		size = list(x.size())
		size[1] = 1
		
		std = torch.std(x, dim=0)
		mean = torch.mean(std)
		return torch.cat((x, mean.repeat(size)),dim=1)
	
class Pixel_norm(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, a):
		b = a / torch.sqrt(torch.sum(a**2, dim=1, keepdim=True)+ 10e-8)
		return b
	
class FromRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		
	def forward(self, x):
		x = self.conv(x)
		return self.relu(x)

class ToRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=(1, 1))
	
	def forward(self, x):

		return self.conv(x)

class G_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()
		if initial_block:
			self.upsample = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 9), stride=(1, 1), padding=(3, 8))
		else:
			self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		self.pixelwisenorm = Pixel_norm()
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)

	def forward(self, x):

		if self.upsample is not None:
			x = self.upsample(x)
		# x = self.conv1(x*scale1)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		# x = self.conv2(x*scale2)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		return x

class D_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()

		self.initial_block = initial_block
		if initial_block:
			self.minibatchstd = Minibatch_std()
			self.conv1 = EqualizedLR_Conv2d(in_ch+1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 9), stride=(1, 1))
			self.outlayer = nn.Sequential(
									nn.Flatten(),
									nn.Linear(out_ch, 1)
									)
			self.auxlayer = nn.Sequential(
									nn.Flatten(),
									nn.Linear(out_ch, 1)
									)
		else:			
			self.minibatchstd = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.relu = nn.LeakyReLU(0.2)
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)
	
	def forward(self, x):
		if self.minibatchstd is not None:
			x = self.minibatchstd(x)
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		#x = self.outlayer(x)
		if self.initial_block:
			return self.outlayer(x), self.auxlayer(x)
		return self.outlayer(x)

class PGGAN_Generator(nn.Module):
	def __init__(self, Y=cfg.model.Y, ngf=cfg.model.ngf, latent_size=cfg.model.nz, 
	      		 min_conc=cfg.data.min_concentration, max_conc=cfg.data.max_concentration, embedding_dim=cfg.model.embedding_dim):
		super().__init__()
		self.depth = 1
		self.alpha = 1
		self.fade_iters = 0
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.current_net = nn.ModuleList([G_Block(latent_size, int(3*ngf/4), initial_block=True)])
		self.toRGBs = nn.ModuleList([ToRGB(ngf, 1)])
		# __add_layers(out_res)
		for d in range(2, 8):
			if d < 6:
				## low res blocks 8x8, 16x16, 32x32 with 512 channels
				in_ch, out_ch = ngf, ngf
			else:
				## from 64x64(5th block), the number of channels halved for each block
				in_ch, out_ch = int(ngf / 2**(d - 6)), int(ngf / 2**(d - 5))
			self.current_net.append(G_Block(in_ch, out_ch))
			self.toRGBs.append(ToRGB(out_ch, 1))
		
		self.Y = Y
		self.nz = latent_size
		self.min_conc = min_conc
		self.max_conc = max_conc
		self.embed_cond = nn.Sequential(
							collections.OrderedDict(
								[ # input is Z, going into a convolution
									("linear0_c", nn.Linear(1, embedding_dim)),
									("relu0_c", nn.LeakyReLU(0.2, inplace=True)),
									("linear1_c", nn.Linear(embedding_dim, embedding_dim*4)),
									("relu1_c", nn.LeakyReLU(0.2, inplace=True)),
									("linear2_c", nn.Linear(embedding_dim*4, int(ngf/4)*4*9)),
									("unflatten_c", nn.Unflatten(1, (int(ngf/4), 4, 9)))
								]
							)
						)

	def forward(self, x, cond):
		if self.Y == "concentration":
			cond = 10*(cond-self.min_conc)/(self.max_conc - self.min_conc)
		cond = self.embed_cond(cond)
		x = self.current_net[0](x)
		if self.depth==1:
			out = torch.concat([x, cond], dim=1)
		else:
			x = torch.concat([x, cond], dim=1)
		for i, block in enumerate(self.current_net[1:self.depth], 1):
			if i == (self.depth-1):
				out = block(x)
			else:
				x = block(x)
		x_rgb = self.toRGBs[self.depth-1](out)
		if self.alpha < 1:
			x_old = self.upsample(x)
			old_rgb = self.toRGBs[self.depth-2](x_old)
			x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb

			self.alpha += self.fade_iters

		return x_rgb
		

	def growing_net(self, num_iters):
		
		self.fade_iters = 1/num_iters
		self.alpha = 1/num_iters

		self.depth += 1


class PGGAN_Discriminator(nn.Module):
	def __init__(self, ndf=cfg.model.ndf):
		super().__init__()
		self.depth = 1
		self.alpha = 1
		self.fade_iters = 0

		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.current_net = nn.ModuleList([D_Block(ndf, ndf, initial_block=True)])
		self.fromRGBs = nn.ModuleList([FromRGB(1, ndf)])

		for d in range(2, 8):
			if d < 6:
				in_ch, out_ch = ndf, ndf
			else:
				in_ch, out_ch = int(ndf / 2**(d - 5)), int(ndf / 2**(d - 6))

			self.current_net.append(D_Block(in_ch, out_ch))
			self.fromRGBs.append(FromRGB(1, in_ch))
	
	def forward(self, x_rgb):
		x = self.fromRGBs[self.depth-1](x_rgb)

		x = self.current_net[self.depth-1](x)

		if self.alpha < 1:
			x_rgb = self.downsample(x_rgb)
			x_old = self.fromRGBs[self.depth-2](x_rgb)
			x = (1-self.alpha)* x_old + self.alpha * x
			self.alpha += self.fade_iters

		for block in reversed(self.current_net[:self.depth-1]):
			x = block(x)
		return x
		
	def growing_net(self, num_iters):

		self.fade_iters = 1/num_iters
		self.alpha = 1/num_iters

		self.depth += 1
