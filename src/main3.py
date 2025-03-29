#!/usr/bin/env python3
import os
import torch
import perceval as pcvl
import perceval.providers.scaleway as scw
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pickle

# Import configuration, models, training routines, and utilities.
from config import load_config, create_scaleway_session
from boson_sampler import BosonSampler, VariationalBosonSampler, ConvolutionalInterferometerBuilder, TriangularInterferometerBuilder, RectangularInterferometerBuilder, PdfInterferometerBuilder, BaseInterferometerBuilder
from model import MnistModel, HybridMnistModel
from training import fit, evaluate
from utils import MNIST_partial, get_dataloader, plot_training_metrics, compute_downsample_shape, pca_downsample
from main_classic import plot_confusion_matrix

# For classification report and LaTeX generation (from main_classic.py)
from sklearn.metrics import classification_report
from datetime import datetime


# Path to PCA model if needed.
# Get the directory of this file (training.py)
base_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level (assuming your project structure has 'src' and 'data' as siblings)
data_dir = os.path.join(base_dir, "..", "data")
pca_model_path = os.path.join(data_dir, "pca_model.pkl")


# Updated plotting function for misclassified predictions.
def plot_misclassified_predictions(model, val_loader, device, result_dir=".", pca=None, bs=None, cfg=None):
    model.eval()
    all_imgs, all_preds, all_labels = [], [], []
    # Determine target downsample shape if quantum embedding is needed.
    target_shape = compute_downsample_shape(bs.nb_parameters) if (bs is not None and cfg is not None) else None
    
    import torch.nn.functional as F
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if model.embedding_size and bs is not None and cfg is not None:
                batch_embeddings = []
                for i in range(imgs.size(0)):
                    img = imgs[i]
                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        img_resized = pca_downsample(img, pca_model)
                    else:
                        img_expanded = img.unsqueeze(0)
                        img_resized = F.interpolate(img_expanded, size=target_shape, mode='bilinear', align_corners=False)
                        img_resized = img_resized.squeeze(0)
                    emb = bs.embed(img_resized, bs.n_samples)
                    batch_embeddings.append(emb)
                emb_tensor = torch.stack(batch_embeddings)
                outputs = model(imgs, emb_tensor)
            else:
                outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_imgs.append(imgs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_imgs = torch.cat(all_imgs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Identify misclassified images.
    mis_idx = (all_preds != all_labels).nonzero(as_tuple=True)[0]
    if mis_idx.numel() == 0:
        print("No misclassified images to plot.")
        return
    # Limit number of misclassified images.
    max_images = 50
    if mis_idx.numel() > max_images:
        mis_idx = mis_idx[:max_images]
    mis_imgs = all_imgs[mis_idx]
    mis_preds = all_preds[mis_idx]
    mis_labels = all_labels[mis_idx]
    
    # Generate collage plot.
    import matplotlib.pyplot as plt
    N = mis_imgs.size(0)
    n_cols = 5
    n_rows = (N + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(N):
        ax = axes[i]
        # Assuming images are in [1,28,28] format.
        ax.imshow(mis_imgs[i].squeeze(0), cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{mis_labels[i].item()} P:{mis_preds[i].item()}", fontsize=8)
    for j in range(N, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    collage_path = os.path.join(result_dir, "misclassified_predictions.png")
    plt.savefig(collage_path, dpi=300)
    plt.close()
    print("Misclassified predictions collage saved to", collage_path)







def setup_result_directory(prefix="hybrid_run"):
    """Create a timestamped directory to save all outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{prefix}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def collect_predictions(model, val_loader, device, bs, cfg):
    """
    Iterate over the validation set to collect true and predicted labels.
    For hybrid models, compute quantum embeddings per image (downsampled as configured).
    """
    model.eval()
    all_preds = []
    all_labels = []
    # Determine target downsample shape.
    target_shape = compute_downsample_shape(bs.nb_parameters)

    
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch  # images shape: [B, 1, 28, 28]
            images, labels = images.to(device), labels.to(device)
            if model.embedding_size:
                batch_embeddings = []
                for i in range(images.size(0)):
                    img = images[i]  # [1,28,28]
                    # Downsample image if necessary:
                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        img_resized = pca_downsample(img, pca_model)
                    else:
                        img_expanded = img.unsqueeze(0)  # [1,1,28,28]
                        img_resized = F.interpolate(img_expanded, size=target_shape, mode='bilinear', align_corners=False)
                        img_resized = img_resized.squeeze(0)
                    # Compute embedding (using standard embed here; adaptive option can be added similarly)
                    emb = bs.embed(img_resized, bs.n_samples)
                    batch_embeddings.append(emb)
                emb_tensor = torch.stack(batch_embeddings)
                outputs = model(images, emb_tensor)
            else:
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def improved_main():
    # 1. Load configuration.
    cfg = load_config()
    print("Loaded configuration:", cfg)

    # 2. Create a results directory.
    result_dir = setup_result_directory(prefix="hybrid_run")
    print(f"Results will be saved to: {result_dir}")

    # 3. Create Scaleway session if remote backend is chosen.
    session = None
    if cfg["quantum"]["backend"] == "scaleway":
        session = create_scaleway_session(cfg)
        session.start()
        print("Scaleway QPU session started.\n")
    else:
        print("Proceeding with local simulation backend.")

    # 4. Select circuit builder.
    circuit_type = cfg["quantum"].get("circuit_type", "triangular")
    if circuit_type == "pdf":
        builder = PdfInterferometerBuilder()
    elif circuit_type == "base":
        builder = BaseInterferometerBuilder()
    elif circuit_type == "triangular":
        builder = TriangularInterferometerBuilder()
    elif circuit_type == "rectangular":
        builder = RectangularInterferometerBuilder()
    elif circuit_type == "convolutional":
        import numpy as np
        filterA = np.random.rand(6, 5)
        filterB = np.random.rand(6, 5)
        builder = ConvolutionalInterferometerBuilder(filterA, filterB, image_size=(28,28))
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    # 5. Initialize BosonSampler.
    if cfg["quantum"].get("variational", False):
        bs = VariationalBosonSampler(
            m=cfg["quantum"]["n_modes"],
            n=cfg["quantum"]["n_photons"],
            postselect=cfg["quantum"]["postselect"],
            backend=cfg["quantum"]["local_backend"],
            session=session,
            builder=builder,
            cache_enabled=cfg["quantum"].get("cache_enabled", False),
            cache_directory=cfg["quantum"].get("cache_directory", "results/cache")
        )
    else:
        bs = BosonSampler(
            m=cfg["quantum"]["n_modes"],
            n=cfg["quantum"]["n_photons"],
            postselect=cfg["quantum"]["postselect"],
            backend=cfg["quantum"]["local_backend"],
            session=session,
            builder=builder,
            cache_enabled=cfg["quantum"].get("cache_enabled", False),
            cache_directory=cfg["quantum"].get("cache_directory", "results/cache")
        )
    bs.n_samples = cfg["quantum"]["n_samples"]
    print(f"BosonSampler: m={bs.m}, n={bs.n}, nb_parameters={bs.nb_parameters}, embedding_size={bs.embedding_size}")
    # Display the circuit layout.
    pcvl.pdisplay(bs.create_circuit())

    # 6. Load MNIST dataset.
    train_dataset = MNIST_partial(data="data", split="train")
    val_dataset = MNIST_partial(data="data", split="val")
    train_loader = get_dataloader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=cfg["data"]["shuffle"])
    val_loader = get_dataloader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 7. Set device.
    device = "cuda" if (cfg["training"]["device"]=="auto" and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    # 8. Initialize model.
    embedding_size = bs.embedding_size if cfg["model"]["type"]=="hybrid" else 0
    if cfg["model"]["type"] == "hybrid":
        model = HybridMnistModel(device=device, embedding_size=embedding_size)
    else:
        model = MnistModel(device=device, embedding_size=0)
    model = model.to(device)
    print(f"Initialized {'quantum-enhanced' if model.embedding_size else 'classical'} model on {device}.")

    # 9. Setup optimizer.
    optimizer_name = cfg["training"]["optimizer"].lower()
    if cfg["quantum"].get("variational", False):
        params_to_opt = list(model.parameters()) + [bs.params]
    else:
        params_to_opt = model.parameters()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params_to_opt, lr=cfg["training"]["learning_rate"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params_to_opt, lr=cfg["training"]["learning_rate"])
    else:
        raise ValueError("Unsupported optimizer specified in config.")

    # 10. Train the model.
    print(f"Starting training for {cfg['training']['epochs']} epochs (lr = {cfg['training']['learning_rate']})...")
    history = fit(
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        bs=bs,
        optimizer=optimizer,
        cfg=cfg
    )
    print("Training completed.")

    # 11. Evaluate the model and collect predictions.
    final_metrics = evaluate(model, val_loader, bs, cfg)
    print(f"Final Validation Loss: {final_metrics['val_loss']:.4f}, Accuracy: {final_metrics['val_acc']:.4f}")

    true_labels, pred_labels = collect_predictions(model, val_loader, device, bs, cfg)
    # Save classification report.
    report = classification_report(true_labels, pred_labels, digits=4)
    report_path = os.path.join(result_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # 12. Generate additional plots and LaTeX reports.
    # Plot training curves (if your fit() function internally plots and saves, this step may be redundant).
    # Here, we assume plot_training_metrics() saves a file "training_curves.png" in the current directory.
    plot_training_metrics([h["val_acc"] for h in history], [h["val_acc"] for h in history],
                            [h["val_loss"] for h in history], [h["val_loss"] for h in history])
    os.replace("training_curves.png", os.path.join(result_dir, "training_curves.png"))
    print("Training curves saved.")

    # Generate and save confusion matrix plot.
    plot_confusion_matrix(true_labels, pred_labels, result_dir)
    print("Confusion matrix plot saved.")

    # Generate and save misclassified predictions collage.
    plot_misclassified_predictions(model, val_loader, device, result_dir, pca=None, bs=bs, cfg=cfg)
    print("Misclassified predictions collage saved.")

    # Optionally, generate LaTeX table and full report.
    try:
        from utils import generate_latex_table, generate_latex_report
        # For demonstration, we use final validation metrics as both training and validation metrics.
        latex_table = generate_latex_table(
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            result_dir
        )
        latex_report = generate_latex_report(
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {"accuracy": final_metrics["val_acc"], "f1_macro": 0.0, "f1_weighted": 0.0},
            {},  # No PCA metrics in the hybrid model case
            result_dir
        )
        print("LaTeX table and report generated.")
    except Exception as e:
        print("Error generating LaTeX reports:", e)

    # 13. Save a summary of results.
    summary = {
        'final_val_loss': final_metrics['val_loss'],
        'final_val_accuracy': final_metrics['val_acc'],
        'training_history': history,
        'classification_report': report
    }
    summary_path = os.path.join(result_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")

    # 14. Stop Scaleway session if used.
    if session:
        session.stop()
        print("Scaleway QPU session stopped.")

if __name__ == "__main__":
    improved_main()
