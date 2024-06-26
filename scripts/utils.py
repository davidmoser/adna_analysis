

def calculate_loss(model, dataloader, loss_function):
    model.eval()
    loss = 0
    weight = 0
    for features, labels in dataloader:
        output = model(features)
        loss += loss_function(output, labels).item() * len(features)
        weight += len(features)
    return loss / weight