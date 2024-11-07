# Welcome to the First Perceval Quest!

Your challenge is to tackle the well-known MNIST problem using a hybrid quantum model on a subset of the original dataset.

The MNIST dataset consists of 70,000 handwritten digit images, each 28x28 pixels. For this quest, you'll work with a reduced dataset of 6,000 images and use a quantum kernel to predict the digits.
## Organization of the repository
The dataset is located in the `data` folder, containing `train.csv` and `test.csv` files. 
The notebook `MNIST_classification_quantum.ipynb` and its equivalent script, `training.py`, contain the training loop used for model training. The code for building quantum embeddings and integrating them into a basic classical model is split in separate scripts: the model is defined in `model.py`, the Boson Sampler in `boson_sampler.py` and some helper functions (dataset class for the the reduced dataset, accuracy function...) can be found in `utils.py`. 

## Challenge Rules

Use any classical machine learning model and demonstrate improved performance with the quantum model (see Evaluation Criteria).
Submit your solution as a reproducible Jupyter notebook.
Modify the provided quantum model as needed. It can rely on quantum kernels or other methods. 

## Challenge Structure

The challenge consists of two phases:

### Phase 1 

Participants submit an initial solution relying on the provided quantum model. The top 10 solutions will advance to Phase 2 based on:

- Accuracy improvements
- Creativity in approach
- Report quality

Selected participants will receive credits for Scaleway GPU simulators to develop extended solutions in Phase 2.

### Phase 2 

Qualified participants will further develop and submit enhanced solutions.

## How to Participate

Email perceval-challenge@quandela.com with your team description (individual or group entries welcome). You'll receive confirmation and submission instructions.

## Support

For any general questions, please use Perceval Forum at https://perceval.quandela.net/forum/ with the tag `Perceval Quest`.
For technical questions, please use the GitHub Discussions tab in this repository.

## Submission Requirements

Your submission must include:
1. Complete code
2. A brief report covering:
   - The classical model description
   - The quantum model implementation such as quantum kernels
   - Comparative results (with/without quantum model)
   - Training duration metrics
   - Any additional relevant insights



## Evaluation Criteria

Solutions will be evaluated against classical models based on:
- Accuracy improvement
- Convergence speed
- Model size optimization at equivalent accuracy

Bonus points for:
- Comprehensive benchmark of the proposed solution against a comparable, state-of-the-art classical approach.
- Successful QPU validation
- Creative approaches

## Timeline

- Registration deadline: November 21st, 2024
- Phase 1 submission daedline: January 6th 2025, 8pm CET
- Phase 2 participant announcement: January 10th 2025
- Phase 2 submission deadline: March 14th 2025, 8pm CET
- Winners announcement: March 21st 2025

## Prizes

Prize details will be announced by November 15th, 2024.

Winners will:
- Receive announced prizes
- Present their solution in a Perceval webinar
- Might contribute to a scientific publication in collaboration with Quandela

Good luck! We look forward to your submissions.
