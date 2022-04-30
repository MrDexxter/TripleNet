import torch
import numpy as np
def train(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs,
	cuda, log_interval):
	for epoch in range(0, n_epochs):
		model.train()
		losses = []
		total_loss = 0
		for batch_idx, (data, target) in enumerate(train_loader):
			#target = target.cuda()
			optimizer.zero_grad()
			outputs = model(data[0],data[1], data[2])
			loss = loss_fn(outputs[0],outputs[1], outputs[2])
			losses.append(loss.item())
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			if batch_idx % log_interval == 0:
				message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				batch_idx * len(data[0]), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), np.mean(losses))
				print(message)
				losses = []
		total_loss /= (batch_idx + 1)
		scheduler.step()
		message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1,
		n_epochs, total_loss)
		val_loss = test_epoch(val_loader, model, loss_fn, cuda)
		val_loss /= len(val_loader)
		message += '\nEpoch: {}/{}. Validation set: Average loss:{:.4f}'.format(epoch + 1, n_epochs, val_loss)
		print(message)
def test_epoch(val_loader, model, loss_fn, cuda):
	with torch.no_grad():
		model.eval()
		val_loss = 0
		for batch_idx, (data, target) in enumerate(val_loader):
			outputs = model(data[0],data[1], data[2])
			loss_outputs = loss_fn(outputs[0],outputs[1],outputs[2])
			val_loss += loss_outputs.item()
	return val_loss