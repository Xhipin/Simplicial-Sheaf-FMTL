import torch
import numpy as np
from torch.utils.data import DataLoader

def evaluate_accuracy(model, dataset, batch_size=64):
    """Evaluate accuracy of a single model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data, targets in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total if total > 0 else 0

def evaluate_all_clients(models, test_datasets):
    """Evaluate all client models and return average accuracy"""
    accuracies = []
    for model, dataset in zip(models, test_datasets):
        acc = evaluate_accuracy(model, dataset)
        accuracies.append(acc)
    return np.mean(accuracies), accuracies

def calculate_communication_bits(graph, factor, num_params, bits_per_param=32):
    """Calculate communication bits for one round"""
    total_bits = 0
    max_neighbors = max(len(list(graph.neighbors(node))) for node in graph.nodes())
    total_bits += 2 * max_neighbors * int(factor * num_params) * bits_per_param
    return total_bits

def count_model_parameters(model):
    """Count the number of parameters in a model"""
    return sum(param.numel() for param in model.parameters())
