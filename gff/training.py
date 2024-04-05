import torch


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def train_epoch(model, optim: torch.optim.Optimizer, dataloader, device):
    for example in dataloader:
        out = recursive_todevice(example, model.device)
        out = model(example)

        optim.zero_grad()


def training_loop(C, model, dataloaders):
    optim = torch.optim.AdamW(model.parameters(), lr=C["lr"])
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optim, lambda epoch: C["lr_decay"] ** epoch
    )

    train_dl, test_dl = dataloaders

    for epoch in range(C["epochs"]):
        train_loss = train_epoch(model, optim, train_dl)
        test_loss = test_epoch(model, optim, test_dl, train=False)

    pass
