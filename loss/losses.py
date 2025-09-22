import torch
import torch.nn.functional as F
from torch import nn

class cross_entropy(nn.Module):
    def __init__(self, weight=None, reduction='mean',ignore_index=256):
        super(cross_entropy, self).__init__()
        self.weight = weight
        self.ignore_index =ignore_index
        self.reduction = reduction


    def forward(self,input, target):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

        return F.cross_entropy(input=input, target=target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=256):
        super(WeightedCrossEntropy, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        target = target.long()
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if input.shape[-1] != target.shape[-1]:
            input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

        # Calculate weights dynamically based on the target
        if self.weight is None:
            # Calculate the pixel-wise weight based on the target
            class_counts = torch.bincount(target.view(-1), minlength=input.size(1)).float()
            class_weights = 1.0 / (class_counts + 1e-6)  # Add small value to avoid division by zero
            class_weights = class_weights / class_weights.sum()  # Normalize weights to sum to 1

            # Apply the weights to the cross-entropy loss
            loss = F.cross_entropy(input=input, target=target, weight=class_weights,
                                   ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            loss = F.cross_entropy(input=input, target=target, weight=self.weight,
                                   ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
    
import time


    
input = torch.randn(16, 3, 256, 256)  # Example input tensor with 2 classes
target = torch.randint(0, 3, (16, 256, 256))  # Example target tensor
print(target.shape)

criterion = WeightedCrossEntropy()


start_time = time.time()
loss = criterion(input, target)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

print(loss)