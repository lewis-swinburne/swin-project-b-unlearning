import torch

orig = torch.load("highway_fast_dqn_normal/model.zip")
unlearn = torch.load("highway_fast_dqn_unlearn/model.zip")

same = True
for k in orig.keys():
    if not torch.equal(orig[k], unlearn[k]):
        same = False
        break

print("Are models identical?", same)
