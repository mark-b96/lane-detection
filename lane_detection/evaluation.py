import torch


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, model, test_dataset, criterion, batch_size: int):
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for image, label in test_loader:
                pred = model(image)
                test_loss += criterion(pred, label).item()
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")