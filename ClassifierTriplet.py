import torch
import torch.nn as nn


class ClassifierTriplet(nn.Module):
    """This is the class made for aggregating a classifier model above a triplet model."""
    def __init__(self):
         super(ClassifierTriplet, self).__init__()
         device = 'cuda' if torch.cuda.is_available() else 'cpu'
         # load already trained Siamese network
         self.model_triplet = torch.load('finaltriplet.tar') # This line loads the trained triplet model from the directory
         # freeze weights and biases of Siamese network
         for param in self.model_triplet.parameters(): # We don't need to learn the paramater for triplet model so we are setting their learning to False
            param.requires_grad = False
         self.classifier = nn.Sequential(
         nn.Linear(2, 50), # first number should match the latent dimension
         nn.ReLU(), # i.e., embedding from Siamese
         nn.Linear(50, 10),
         nn.Softmax(dim=1)
         ).to(device)  # This is a model defined that will be appended on to the triplet model. It classifies the data into 10 classes.
         for param in self.classifier.parameters():  # Setting the parameters of the classifier model to Learnable
            param.requires_grad = True
    def forward(self,x):
         h = self.model_triplet.get_embedding(x)  # This is one step implementation when an input is fed into the classifier model.
                                                  # It will just get the embedding for the current input and return it after feeding it to the classifier
         return self.classifier(h)