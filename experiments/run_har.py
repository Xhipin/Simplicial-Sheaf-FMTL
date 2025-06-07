import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import copy
import time
import json
from torch.utils.data import DataLoader

from datasets.har import HARDataset
from models.linear import MultinomialLogisticRegression
from algorithms.sheaf_fmtl import SheafFMTL
from utils.graph_utils import generate_graph_by_type, visualize_graph, get_graph_statistics
from utils.metrics import evaluate_all_clients, calculate_communication_bits, count_model_parameters

def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare dataset
    print("Preparing HAR dataset...")
    dataset = HARDataset(
        num_clients=args.num_clients,
        downsample_rate=args.downsample_rate,
        task_index_path=args.task_index_path
    )
    client_train_datasets, client_test_datasets = dataset.prepare_data()
    
    # Get dataset info
    data_info = dataset.get_data_info()
    input_size = data_info['input_size']
    num_classes = data_info['num_classes']
    actual_num_clients = data_info['num_clients']
    
    print(f"Dataset info: {data_info}")
    
    # Generate communication graph
    print(f"Generating {args.graph_type} communication graph...")
    graph = generate_graph_by_type(
        actual_num_clients, 
        graph_type=args.graph_type,
        edge_probability=args.edge_probability,
        seed=args.seed
    )
    
    # Print graph statistics
    stats = get_graph_statistics(graph)
    print(f"Graph statistics: {stats}")
    
    # Visualize graph
    if args.visualize_graph:
        visualize_graph(graph, title=f"HAR - {args.graph_type} Graph")
    
    # Initialize models
    print("Initializing client models...")
    base_model = MultinomialLogisticRegression(input_size, num_classes)
    client_models = [copy.deepcopy(base_model) for _ in range(actual_num_clients)]
    
    # Create data loaders
    train_loaders = [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for dataset in client_train_datasets
    ]
    
    # Initialize Sheaf-FMTL algorithm
    print("Initializing Sheaf-FMTL algorithm...")
    sheaf_fmtl = SheafFMTL(
        models=client_models,
        graph=graph,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        eta=args.eta,
        gamma=args.gamma
    )
    
    # Training metrics
    history = {
        'test_accuracy': [],
        'communication_bits': [],
        'cpu_time': []
    }
    
    cumulative_bits = 0
    cumulative_time = 0
    num_params = count_model_parameters(client_models[0])
    bits_per_round = calculate_communication_bits(graph, args.gamma, num_params)
    
    print(f"\nStarting training for {args.num_rounds} rounds...")
    print(f"Communication bits per round: {bits_per_round:,}")
    print(f"Number of parameters: {num_params}")
    
    # Training loop
    for round_idx in range(args.num_rounds):
        round_start_time = time.time()
        
        # Train all clients
        for client_id in range(actual_num_clients):
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
        cumulative_bits += bits_per_round
        round_time = time.time() - round_start_time
        cumulative_time += round_time
        
        # Evaluate
        avg_accuracy, client_accuracies = evaluate_all_clients(
            client_models, client_test_datasets
        )
        
        history['test_accuracy'].append(avg_accuracy)
        history['communication_bits'].append(cumulative_bits)
        history['cpu_time'].append(cumulative_time)
        
        # Print progress
        if round_idx % args.print_every == 0:
            print(f"Round {round_idx:3d}: "
                  f"Avg Test Accuracy = {avg_accuracy:.4f}, "
                  f"Bits = {cumulative_bits/1e6:.2f} MB, "
                  f"Time = {cumulative_time:.2f}s")
    
    # Save results
    results = {
        'args': vars(args),
        'dataset_info': data_info,
        'graph_stats': stats,
        'history': history,
        'final_accuracy': history['test_accuracy'][-1],
        'total_communication_mb': cumulative_bits / 1e6,
        'total_time_seconds': cumulative_time
    }
    
    if args.save_results:
        save_path = f"results/sheaf_fmtl_har_gamma{args.gamma}.json"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    print(f"\nTraining completed!")
    print(f"Final average test accuracy: {history['test_accuracy'][-1]:.4f}")
    print(f"Total communication: {cumulative_bits/1e6:.2f} MB")
    print(f"Total time: {cumulative_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sheaf-FMTL on HAR Dataset")
    
    # Dataset parameters
    parser.add_argument('--num_clients', type=int, default=30, help='Number of clients')
    parser.add_argument('--downsample_rate', type=float, default=0.2, 
                        help='Downsample rate for half of the clients')
    parser.add_argument('--task_index_path', type=str, default='data/task_index.npy',
                        help='Path to task index file')
    
    # Model parameters
    parser.add_argument('--lambda_reg', type=float, default=0.05, help='Regularization parameter')
    parser.add_argument('--alpha', type=float, default=0.005, help='Learning rate for models')
    parser.add_argument('--eta', type=float, default=0.01, help='Learning rate for restriction maps')
    parser.add_argument('--gamma', type=float, default=0.3, help='Compression factor for interaction space')
    
    # Training parameters
    parser.add_argument('--num_rounds', type=int, default=200, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--l2_strength', type=float, default=0.01, help='L2 regularization strength')
    
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
