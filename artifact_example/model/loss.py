from torch import nn

def loss_fn(outputs, labels):
    """
    Thi is example code.
    Customize your self.
    """
    return nn.CrossEntropyLoss()(outputs, labels)