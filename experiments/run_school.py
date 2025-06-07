import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import copy
import time
import json
from torch.utils.data import DataLoader

from datasets.school import SchoolDataset
from models.linear import LinearRegression
from algorithms.sheaf_fmtl import SheafFMTL
from utils.graph_utils import generate_graph_by_type, visualize_graph, get_graph_statistics
from utils.metrics import count_model_parameters

def evaluate_mse(model, dataset, batch_size=32):
    """Evaluate MSE for regression task"""
    model.eval()
    total_loss = 0
    total_count = 0
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data, targets in dataloader:
            predictions = model(data)
            loss = nn.MSELoss()(predictions, targets.view(-1, 1))
            total_loss += loss.item() * data.size(0)
            total_count += data.size(0)
    
    return total_loss / total_count if total_count > 0 else float('inf')

def evaluate_all_clients_mse(models, test_datasets):
    """Evaluate all client models and return average MSE"""
    mses = []
    for model, dataset in zip(models, test_datasets):
        if len(dataset) > 0:  # Only evaluate if client has test data
            mse = evaluate_mse(model, dataset)
            mses.append(mse)
    return np.mean(mses) if mses else float('inf'), mses

class SheafFMTLRegression(SheafFMTL):
    """Modified Sheaf-FMTL for regression tasks"""
    
    def local_update(self, client_id, dataloader, local_epochs=1, l2_strength=0.5):
        """Perform local training on client's data for regression"""
        model = self.models[client_id]
        optimizer = torch.optim.SGD(model.parameters(), lr=self.alpha)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(local_epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                
                # Add L2 regularization
                l2_reg = sum(param.pow(2).sum() for param in model.parameters())
                loss = loss + l2_strength * l2_reg
                
                loss.backward()
                optimizer.step()

def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare dataset
    print("Preparing School dataset...")
    dataset = SchoolDataset(
        data_path=args.data_path,
        bias=args.bias,
        standardize=args.standardize,
        seed=args.seed
    )
    client_train_datasets, client_test_datasets = dataset.prepare_data()
    
    # Get dataset info
    data_info = dataset.get_data_info()
    input_size = data_info['input_size']
    output_size = data_info['output_size']
    num_clients = data_info['num_clients']
    
    print(f"Dataset info: {data_info}")
    
    # Generate communication graph
    print(f"Generating {args.graph_type} communication graph...")
    graph = generate_graph_by_type(
        num_clients, 
        graph_type=args.graph_type,
        edge_probability=args.edge_probability,
        seed=args.seed
    )
    
    # Print graph statistics
    stats = get_graph_statistics(graph)
    print(f"Graph statistics: {stats}")
    
    # Visualize graph (note: might be crowded with 139 nodes)
    if args.visualize_graph:
        visualize_graph(graph, title=f"School - {args.graph_type} Graph")
    
    # Initialize models
    print("Initializing client models...")
    base_model = LinearRegression(input_size, output_size, bias=True)
    client_models = [copy.deepcopy(base_model) for _ in range(num_clients)]
    
    # Create data loaders
    train_loaders = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=True) if len(dataset) > 0 
        else DataLoader(dataset, batch_size=1, shuffle=False)  # Handle empty datasets
        for dataset in client_train_datasets
    ]
    
    # Initialize Sheaf-FMTL algorithm (regression version)
    print("Initializing Sheaf-FMTL algorithm for regression...")
    sheaf_fmtl = SheafFMTLRegression(
        models=client_models,
        graph=graph,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        eta=args.eta,
        gamma=args.gamma
    )
    
    # Training metrics
    history = {
        'test_mse': [],
        'communication_bits': [],
        'cpu_time': []
    }
    
    cumulative_bits = 0
    cumulative_time = 0
    num_params = count_model_parameters(client_models[0])
    
    # Calculate communication cost per round
    total_bits_per_round = 0
    for node in graph.nodes():
        num_neighbors = len(list(graph.neighbors(node)))
        total_bits_per_round += 2 * num_neighbors * int(args.gamma * num_params) * 32
    
    print(f"\nStarting training for {args.num_rounds} rounds...")
    print(f"Communication bits per round: {total_bits_per_round:,}")
    print(f"Number of parameters: {num_params}")
    
    # Training loop
    for round_idx in range(args.num_rounds):
        round_start_time = time.time()
        
        # Train all clients
        for client_id in range(num_clients):
            if len(client_train_datasets[client_id]) > 0:  # Only train if client has data
                # Local update
                sheaf_fmtl.local_update(
                    client_id, 
                    train_loaders[client_id],
                    local_epochs=args.local_epochs,
                    l2_strength=args.l2_strength
                )
                
                # Sheaf update
                sheaf_fmtl.sheaf_update(client_id)
                
                # Update restriction maps
                sheaf_fmtl.update_restriction_maps(client_id)
        
        # Calculate metrics
        cumulative_bits += total_bits_per_round
        round_time = time.time() - round_start_time
        cumulative_time += round_time
        
        # Evaluate
        avg_mse, client_mses = evaluate_all_clients_mse(
            client_models, client_test_datasets
        )
        
        history['test_mse'].append(avg_mse)
        history['communication_bits'].append(cumulative_bits)
        history['cpu_time'].append(cumulative_time)
        
        # Print progress
        if round_idx % args.print_every == 0:
            print(f"Round {round_idx:3d}: "
                  f"Avg Test MSE = {avg_mse:.6f}, "
                  f"Bits = {cumulative_bits/1e6:.2f} MB, "
                  f"Time = {cumulative_time:.2f}s")
    
    # Save results
    results = {
        'args': vars(args),
        'dataset_info': data_info,
        'graph_stats': stats,
        'history': history,
        'final_mse': history['test_mse'][-1],
        'total_communication_mb': cumulative_bits / 1e6,
        'total_time_seconds': cumulative_time
    }
    
    if args.save_results:
        save_path = f"results/sheaf_fmtl_school_gamma{args.gamma}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    print(f"\nTraining completed!")
    print(f"Final average test MSE: {history['test_mse'][-1]:.6f}")
    print(f"Total communication: {cumulative_bits/1e6:.2f} MB")
    print(f"Total time: {cumulative_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sheaf-FMTL on School Dataset")
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='data/school.mat',
                        help='Path to school dataset')
    parser.add_argument('--bias', action='store_true', default=True,
                        help='Add bias term to features')
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='Standardize features and normalize targets')
    
    # Model parameters
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='Regularization parameter')
    parser.add_argument('--alpha', type=float, default=0.005, help='Learning rate for models')
    parser.add_argument('--eta', type=float, default=0.01, help='Learning rate for restriction maps')
    parser.add_argument('--gamma', type=float, default=0.3, help='Compression factor for interaction space')
    
    # Training parameters
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--l2_strength', type=float, default=0.5, help='L2 regularization strength')
    
    # Graph parameters
    parser.add_argument('--graph_type', type=str, default='erdos_renyi', 
                        choices=['erdos_renyi', 'small_world', 'scale_free', 'complete'],
                        help='Type of communication graph')
    parser.add_argument('--edge_probability', type=float, default=0.2, 
                        help='Edge probability for Erdos-Renyi graph')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--print_every', type=int, default=10, help='Print frequency')
    parser.add_argument('--visualize_graph', action='store_true', help='Visualize communication graph')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    main(args)
