import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_loss(csv_file='/workdir/radish/hachi/Output/experiments/objectstitch/2026-02-24T16-38-42/2026-02-24T16-38-42/csv/metrics.csv', output_folder='plots', output_filename='training_loss.png'):
    # Configuration
    # csv_file =  # ObjectStitch-Image-Composition/experiments/objectstitch/2025-10-20T15-34-53/2025-10-20T15-34-53/csv/metrics.csv
    # output_folder = 'plots'  # Change this to your desired folder
    # output_filename = 'training_loss.png'

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter out rows where train/loss_step is not null
    df_train = df[df['train/loss_step'].notna()].copy()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.plot(df_train['global_step'], df_train['train/loss_step'], 
            label='Training Loss', linewidth=1, alpha=0.7)

    # Add labels and title
    plt.xlabel('Global Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Use log scale for y-axis if loss varies significantly
    plt.yscale('log')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save without displaying
    plt.close()

def plot_training_loss_epoch(csv_file='/workdir/radish/hachi/Output/experiments/objectstitch/2026-02-24T16-38-42/2026-02-24T16-38-42/csv/metrics.csv', output_folder='plots', output_filename='training_loss_epoch.png'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter out rows where train/loss_epoch is not null
    df_train = df[df['train/loss_epoch'].notna()].copy()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot training loss per epoch
    plt.plot(df_train['epoch'], df_train['train/loss_epoch'], 
            label='Training Loss per Epoch', linewidth=2, alpha=0.8)

    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Use log scale for y-axis if loss varies significantly
    plt.yscale('log')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Save without displaying
    plt.close()

plot_training_loss(output_folder="./outputs/plots")
plot_training_loss_epoch(output_folder="./outputs/plots")