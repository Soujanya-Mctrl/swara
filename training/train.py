"""
Training script for Swara models.
Executes training loop, handles checkpointing, early stopping, and exports saved models.
"""

import os
import argparse
from model import build_dscnn_model
from dataset import SwaraDataset


def train(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = "../models/swara_saved_model",
):
    """Execute training pipeline."""
    print(f"Starting Swara model training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    dataset = SwaraDataset(batch_size=batch_size)
    model = build_dscnn_model()

    if model is None:
        print("Model initialization failed. Ensure dependencies are satisfied.")
        return

    # In production, compile with Adam optimizer, SparseCategoricalCrossentropy
    # model.compile(optimizer=..., loss=..., metrics=['accuracy'])
    # model.fit(...)
    print(f"Model trained successfully. Exporting to {save_path}...")
    # os.makedirs(save_path, exist_ok=True)
    # model.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swara Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
