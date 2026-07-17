"""
predict.py
----------
Single-image inference helper: takes a PIL image + a loaded model and
returns the predicted class, confidence, and full class-probability
breakdown.
"""

import torch


def predict_image(model, input_tensor, class_names, device="cpu"):
    """
    Run the model on a single preprocessed (1, 3, H, W) tensor.

    Returns
    -------
    predicted_name : str
    confidence : float           (0-100)
    probabilities : dict[str, float]   class_name -> probability (0-100)
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_index = torch.max(probs, dim=0)
    predicted_name = class_names[predicted_index.item()]

    probabilities = {
        class_names[i]: probs[i].item() * 100
        for i in range(len(class_names))
    }

    return predicted_name, confidence.item() * 100, probabilities
