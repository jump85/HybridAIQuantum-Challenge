# src/main.py
import torch
print(torch.cuda.is_available())
from config import load_config, create_scaleway_session
from boson_sampler import BosonSampler
from model import MnistModel, HybridMnistModel
from training import fit
from utils import MNIST_partial, get_dataloader

def main():
    # 1. Load configuration from default_config.yaml
    cfg = load_config()
    print("Loaded configuration:", cfg)

    # 2. Create Scaleway session if backend is set to remote
    session = None
    if cfg["quantum"]["backend"] == "scaleway":
        session = create_scaleway_session(cfg)
        session.start()
    
    # 3. Initialize BosonSampler
    from boson_sampler import BosonSampler, ConvolutionalInterferometerBuilder, TriangularInterferometerBuilder, RectangularInterferometerBuilder, VariationalBosonSampler, PdfInterferometerBuilder, BaseInterferometerBuilder



    # Read circuit_type from your config (e.g., "pdf" or "alternative")
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
        # Define filters. Here we use random filters of shape (6,5) for demonstration.
        import numpy as np
        filterA = np.random.rand(6, 5)
        filterB = np.random.rand(6, 5)
        builder = ConvolutionalInterferometerBuilder(filterA, filterB, image_size=(28,28))
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

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

    print("BosonSampler embedding size:", bs.embedding_size)

    # 4. Load dataset using MNIST_partial
    train_dataset = MNIST_partial(data="data", split="train")
    val_dataset = MNIST_partial(data="data", split="val")
    train_loader = get_dataloader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=cfg["data"]["shuffle"])
    val_loader = get_dataloader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    # 5. Determine device
    if cfg["training"]["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["training"]["device"]
    print("Using device:", device)

    # 6. Initialize model
    embedding_size = bs.embedding_size if cfg["model"]["type"] == "hybrid" else 0
    if cfg["model"]["type"] == "hybrid":
        net = HybridMnistModel(device=device, embedding_size=embedding_size)
        #net = HybridMnistModel(device=device, embedding_size=embedding_size + cfg["model"].get("fourier_features", 0))
    else:
        net = MnistModel(device=device, embedding_size=0)
    net = net.to(device)

    # 7. Setup optimizer (supports "adam" or "sgd")
    optimizer_name = cfg["training"]["optimizer"].lower()
    if cfg["quantum"].get("variational", False):
        params_to_opt = list(net.parameters()) + [bs.params]
    else:
        params_to_opt = net.parameters()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params_to_opt, lr=cfg["training"]["learning_rate"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params_to_opt, lr=cfg["training"]["learning_rate"])
    else:
        raise ValueError("Unsupported optimizer specified in config.")

    # 8. Train the model
    history = fit(
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["learning_rate"],
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        bs=bs,
        optimizer=optimizer,
        cfg=cfg
    )
    print("Training completed.")

    # 9. Stop Scaleway session if used
    if session:
        session.stop()

if __name__ == "__main__":
    main()
