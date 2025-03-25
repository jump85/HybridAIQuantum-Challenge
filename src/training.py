import torch
from tqdm import tqdm
import time
from utils import plot_training_metrics

def fit(epochs, lr, model, train_loader, val_loader, bs, optimizer, cfg):
    """
    Train the model for a specified number of epochs.
    If the model uses quantum embedding (embedding_size > 0), embed images using the BosonSampler.
    """
    history = []
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if model.embedding_size:
                # Assume batch contains a list of images if batch_size > 1.
                images, labs = batch  # images shape: [B, 1, 28, 28]
                # Remove extra dimensions if needed
                image_list = [img.squeeze() for img in images]  # Each image now shape [28, 28]
                # Generate quantum embedding from the image

                # if cfg["quantum"].get("adaptive_sampling", False):
                #     emb_list = bs.parallel_embed(image_list, bs.n_samples, adaptive=True, min_samples=100, tol=1e-3)
                #     #emb = bs.adaptive_embed(images, min_samples=100, max_samples=bs.n_samples, tol=1e-3).unsqueeze(0)
                #     #emb = bs.adaptive_embed(img_proc, min_samples=100, max_samples=bs.n_samples, tol=1e-3)
                # else:
                #     emb_list = bs.parallel_embed(image_list, bs.n_samples, adaptive=False)
                #     #emb = bs.embed(images, bs.n_samples).unsqueeze(0)
                # #embeddings.append(emb)

                emb_list = []
                for img in image_list:
                    if cfg["quantum"].get("adaptive_sampling", False):
                        emb = bs.adaptive_embed(img, min_samples=100, max_samples=bs.n_samples, tol=1e-3)
                    else:
                        emb = bs.embed(img, bs.n_samples)
                    emb_list.append(emb)


                # Stack embeddings and add the batch dimension if needed.
                #embeddings = torch.stack(embeddings)  # shape: [B, embedding_size]
                emb = torch.stack(emb_list)

                # TODO concatenate Fourier features
                #  # Compute Fourier features for each image and stack them:
                # fourier_feats = [fourier_encode(img, n_features=cfg["model"].get("fourier_features", 50)) for img in image_list]
                # fourier_feats = torch.stack(fourier_feats)  # shape: [B, fourier_features]

                # # Concatenate along feature dimension:
                # emb = torch.cat((emb, fourier_feats), dim=1)  # new embedding size = bs.embedding_size + fourier_features


                # Now call the training step (ensure your model.training_step accepts batched embeddings)
                loss, acc = model.training_step((images, labs), emb=emb)
                #loss, acc = model.training_step(batch, emb=emb)
            else:
                loss, acc = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach())
            running_acc += float(acc.detach())
            if model.embedding_size and step % 100 == 0 and step > 0:
                print(f"Step {step}, Training acc: {running_acc/(step+1):.4f}, Loss: {running_loss/(step+1):.4f}")
        result = evaluate(model, val_loader, bs, cfg)
        model.epoch_end(epoch, result)
        history.append(result)
        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(running_acc / len(train_loader))
        val_loss_list.append(result["val_loss"])
        val_acc_list.append(result["val_acc"])
        plot_training_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    return history



#TODO Integrate Fourier Features into the Data Pipeline  **
#  Concatenate with Quantum Embedding. OR  Preprocess Images Before Feeding into the CNN.
#fourier_feats = fourier_encode(images.squeeze(0).squeeze(0), n_features=50)  # choose 50 features
## After obtaining the quantum embedding (emb), concatenate them:
#emb = torch.cat((emb, fourier_feats.unsqueeze(0)), dim=1)
#######
#fourier_size = cfg["model"].get("fourier_features", 0)
#total_embedding_size = bs.embedding_size + fourier_size if cfg["model"]["type"] == "hybrid" else 0
## When initializing HybridMnistModel:
#net = HybridMnistModel(device=device, embedding_size=total_embedding_size)




def evaluate(model, val_loader, bs, cfg):
    """
    Evaluate the model on the validation set.
    If quantum embedding is used, compute it for each sample.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            if model.embedding_size:
                
                images, labs = batch
                images = images.squeeze(0).squeeze(0)
                if cfg["quantum"].get("adaptive_sampling", False):
                    emb = bs.adaptive_embed(images, min_samples=100, max_samples=bs.n_samples, tol=1e-3).unsqueeze(0)
                else:
                    emb = bs.embed(images, bs.n_samples).unsqueeze(0)
                out = model.validation_step(batch, emb=emb)
            else:
                out = model.validation_step(batch)
            outputs.append(out)
    batch_losses = [x["val_loss"] for x in outputs]
    batch_accs = [x["val_acc"] for x in outputs]
    val_loss = torch.stack(batch_losses).mean().item()
    val_acc = torch.stack(batch_accs).mean().item()
    return {"val_loss": val_loss, "val_acc": val_acc}
