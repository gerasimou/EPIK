# for i, q in enumerate(self.quantiles):
#     errors = target - preds[:, i]
#     losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
#
# loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))


import numpy as np

quants = [0.05, 0.95]

target = [0.2, 0.4, 0.7, 0.9]

preds = np.array(
[[0.3, 0.3, 0.3, 0.3],
[0.8, 0.8, 0.8, 0.8]]
).T


for i, q in enumerate(quants):
    print( i, q )

    errors = target - preds[:, i]
    print(errors)

    # loss = np.max( (q-1)*errors, q * errors)
    loss1 = (q-1)*errors
    loss2 = q * errors
    print(loss1, loss2)

# print(preds[:,1])