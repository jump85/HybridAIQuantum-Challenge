#!/usr/bin/env python3
# main.py - Hybrid quantum-classical ML pipeline for MNIST classification

# Make sure Perceval 0.11.2 is installed. 
# If using Perceval 0.12+, see notes below for adjusting result retrieval.

#### IMPORTS ####
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import perceval as pcvl                          # Photonic quantum computing framework (Quandela Perceval)
import perceval.providers.scaleway as scw        # Scaleway QPU provider (for remote execution)

# Import our modules (provided separately)
from utils import MNIST_partial, plot_training_metrics
from model import MnistModel
from training import evaluate       # evaluate() handles validation, including quantum embedding
from boson_sampler import BosonSampler

#### LOAD DATASETS ####
# Load the reduced MNIST datasets (from CSV files) for training and validation.
# Assumes './data/train.csv' and './data/val.csv' as provided in the challenge.
train_dataset = MNIST_partial(split='train')   # uses default data path './data'
val_dataset   = MNIST_partial(split='val')
# If files are in a different location, specify the `data` parameter, e.g., MNIST_partial(data='/path/to/data', split='train').

# Create DataLoader for each dataset.
# We use batch_size=1 because the BosonSampler currently processes one sample at a time (required for embedding).
batch_size = 130
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

#### SCALEWAY QaaS SESSION (OPTIONAL) ####
# Initialize a Scaleway QPU session if credentials are provided. Otherwise, use local simulation.
session = None
project_id = os.environ.get("SCW_PROJECT_ID", "")
token      = os.environ.get("SCW_SECRET_KEY", "")
if project_id and token:
    # Choose the remote platform for simulation or hardware (default to a GPU simulator).
    platform = os.environ.get("SCW_PLATFORM", "sim:sampling:p100")  # e.g., "sim:sampling:h100" for H100 GPU, or actual QPU if available
    session = scw.Session(project_id=project_id, token=token, platform=platform)
    print(f"Initialized Scaleway session on platform '{platform}'.")
else:
    print("No Scaleway credentials found. Proceeding with local quantum simulation (Perceval will use CPU/GPU as available).")

# Start the remote session if it was created.
if session is not None:
    session.start()
    print("Scaleway QPU session started.\n")
    # (If a session with the same deduplication_id is already running, start() will reuse it.)
    # NOTE: A running session is billed per minute. Always stop the session when done to avoid charges.

#### BOSON SAMPLER DEFINITION ####
# Define the BosonSampler for quantum feature embedding.
# Here we use m=30 modes and n=2 photons, with postselection on detecting 2 photons.
bs = BosonSampler(m=30, n=2, postselect=2, session=session)
print(f"BosonSampler defined with m={bs.m}, n={bs.n}, nb_parameters={bs.nb_parameters}, embedding_size={bs.embedding_size}")

# Display the quantum circuit structure (this will print or render the interferometer layout).
pcvl.pdisplay(bs.create_circuit())

# Note: The BosonSampler uses Perceval's Sampler internally to compute output distributions.
# In Perceval v0.11.2, Sampler.probs() returns a dict with results. If using Perceval v0.12+, it returns a job object, 
# so you would retrieve results with job.get('results') instead of direct dict access (adjust inside boson_sampler if needed).

#### DEVICE SETUP (CPU/GPU) ####
# Determine device to run the torch model on (use GPU if available for faster training).
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#### MODEL INITIALIZATION ####
# Initialize the model. 
# If using the quantum embedding from the BosonSampler, include embedding_size in the model.
model = MnistModel(device=device)  
# To enable hybrid quantum-classical model input, uncomment the line below to concatenate quantum features:
# model = MnistModel(device=device, embedding_size=bs.embedding_size)  
model = model.to(device)
print(f"Initialized {'quantum-enhanced' if model.embedding_size else 'classical'} model on {device}.")

#### TRAINING LOOP DEFINITION ####
def fit(epochs, lr, model, train_loader, val_loader, bs, opt_func=torch.optim.SGD):
    """
    Train the model for a given number of epochs, using the specified optimizer and learning rate.
    This will also evaluate on the validation set at the end of each epoch and plot training curves.
    """
    history = []  # to record validation metrics per epoch
    optimizer = opt_func(model.parameters(), lr)
    # Lists to store metrics for plotting
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    for epoch in range(epochs):
        model.train()  # set model to training mode
        training_loss_sum = 0.0
        training_acc_sum = 0.0
        # --- Training Phase ---
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # If using quantum embedding, compute the embedding via BosonSampler for the current image
            if model.embedding_size:
                images, labels = batch
                # Reshape [1,1,28,28] -> [28,28] to pass into BosonSampler
                images = images.squeeze(0).squeeze(0)
                # Embed the image into a quantum feature vector (distribution of photon detection events)
                embs = bs.embed(images, n_sample=1000)  # use 1000 samples for estimating probabilities
                # Add batch dimension back to embedding
                loss, acc = model.training_step((images.unsqueeze(0).unsqueeze(0), labels), emb=embs.unsqueeze(0))
            else:
                # Classical model (no quantum embedding)
                loss, acc = model.training_step(batch)
            # Backpropagation and weight update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Accumulate training metrics
            training_loss_sum += float(loss.detach())
            training_acc_sum  += float(acc.detach())
            # Print intermediate training metrics for quantum model every 100 steps (to monitor progress)
            if model.embedding_size and step % 100 == 0:
                avg_acc  = training_acc_sum / (step + 1)
                avg_loss = training_loss_sum / (step + 1)
                print(f"  Step {step:3d}: partial train accuracy = {avg_acc:.4f}, loss = {avg_loss:.4f}")
        # --- Validation Phase ---
        model.eval()  # set model to evaluation mode
        result = evaluate(model, val_loader, bs)  # computes val_loss and val_acc over the whole val set
        val_loss, val_acc = result['val_loss'], result['val_acc']
        model.epoch_end(epoch, result)           # print epoch summary (validation loss & accuracy)
        history.append(result)
        # Record epoch metrics for plotting
        train_loss_list.append(training_loss_sum / len(train_loader))
        train_acc_list.append(training_acc_sum / len(train_loader))
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # Plot and save training curves after each epoch
        plot_training_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    return history

#### TRAINING EXECUTION ####
# Set training parameters
epochs = 5
learning_rate = 0.001
print(f"\nStarting training for {epochs} epochs (learning rate = {learning_rate})...")
# Train the model (on local simulator or remote QPU if session is active)
history = fit(epochs, learning_rate, model, train_loader, val_loader, bs)
print("Training completed.")

#### END SESSION ####
# Stop the Scaleway session if it was started, to free resources (avoid additional charges).
if session is not None:
    session.stop()
    print("Scaleway QPU session stopped.")
