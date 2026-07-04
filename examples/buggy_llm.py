import torch


@torch.compile
def full_step(batch):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()  # compiled together with backward -> conflict with gradient_surgery


def train(model, loader, optimizer):
    losses = []
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()          # no zero_grad()
        optimizer.step()
        losses.append(loss)       # retains autograd graph -> memory leak
