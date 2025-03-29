#!/usr/bin/env python3
import os
import torch
import perceval as pcvl
import perceval.providers.scaleway as scw
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import MNIST_partial, get_dataloader, plot_training_metrics
from model import MnistModel  # For a hybrid model, import HybridMnistModel instead
from training import evaluate, fit
from boson_sampler import BosonSampler

def main():
    # --- Data Loading ---
    # Load the reduced MNIST dataset (CSV version)
    train_dataset = MNIST_partial(split='train')
    val_dataset = MNIST_partial(split='val')
    batch_size = 30  # Adjust as needed (if quantum embedding is used, batch size may need to be lower)
    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size, shuffle=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # --- Scaleway QaaS Session (Remote Simulation) ---
    session = None
    project_id = os.environ.get("SCW_PROJECT_ID", "")
    token = os.environ.get("SCW_SECRET_KEY", "")
    if project_id and token:
        platform = os.environ.get("SCW_PLATFORM", "sim:sampling:p100")
        session = scw.Session(project_id=project_id, token=token, platform=platform)
        print(f"Initialized Scaleway session on platform '{platform}'.")
        session.start()
        print("Scaleway QPU session started.\n")
    else:
        print("No Scaleway credentials found. Proceeding with local quantum simulation.")

    # --- BosonSampler Definition ---
    # Here we use example parameters; adjust m, n, postselect as needed.
    bs = BosonSampler(m=30, n=2, postselect=2, session=session)
    print(f"BosonSampler defined with m={bs.m}, n={bs.n}, nb_parameters={bs.nb_parameters}, embedding_size={bs.embedding_size}")
    # Display the interferometer circuit layout
    pcvl.pdisplay(bs.create_circuit())

    # --- Device Setup ---
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Model Initialization ---
    # For a hybrid model with quantum embedding, use HybridMnistModel and pass embedding_size=bs.embedding_size.
    model = MnistModel(device=device)  # Switch to HybridMnistModel if using quantum features.
    model = model.to(device)
    print(f"Initialized {'quantum-enhanced' if model.embedding_size else 'classical'} model on {device}.")

    # --- Training ---
    epochs = 5
    learning_rate = 0.001
    print(f"\nStarting training for {epochs} epochs (learning rate = {learning_rate})...")
    history = fit(epochs, learning_rate, model, train_loader, val_loader, bs, opt_func=torch.optim.Adam)
    print("Training completed.")

    # --- End Session ---
    if session is not None:
        session.stop()
        print("Scaleway QPU session stopped.")

if __name__ == "__main__":
    main()