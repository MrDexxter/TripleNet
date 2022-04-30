
import sys
import torch

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import MNISTDataLoader
import matplotlib.pyplot as plt
def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = torch.load('finaltriplet.tar')
	train_loader, test_loader = MNISTDataLoader.get_loader_normal()
	model.eval()
	inputs, embs, targets = [], [], []
	for x, t in tqdm(test_loader, total=len(test_loader)):
		x = x.to(device)
		t = t.to(device)
		o1 = model(x)
		embs.append(o1.cpu().data.numpy())
		targets.append(t.cpu().numpy())
	
	embed = np.array(embs).reshape((-1, 2)) # outside of for loop
	targets = np.array(targets).reshape((-1,))
	labelset = set(targets.tolist())
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	for label in labelset:
		indices = np.where(targets == label)
		ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
		ax.legend()
		fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
	plt.plot()
	plt.draw()
	plt.show()
if __name__ == "__main__":
	sys.exit(int(main() or 0))