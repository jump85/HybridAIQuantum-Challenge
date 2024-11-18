# Welcome to the First Perceval Quest!

<table align="center" border="0">
<tr>
  <td><img src="https://www.quandela.com/wp-content/themes/quandela/assets/images/quandela-logo.png" width="200" alt="Quandela Logo"></td>
  <td valign="middle" style="font-size: 24px; font-weight: bold; padding: 0 20px;">×</td>
  <td><img src="https://www.scaleway.com/_next/static/media/logo.7e2996cb.svg" width="200" alt="Scaleway Logo"></td>
</tr>
</table>

## About the Challenge

The First Perceval Quest is jointly organized by Quandela and Scaleway to explore the intersection of quantum computing and machine learning through one of the most iconic machine learning benchmarks - the MNIST dataset.

Your challenge is to tackle the well-known MNIST problem using a hybrid quantum model on a subset of the original dataset. The MNIST dataset consists of 70,000 handwritten digit images, each 28x28 pixels. For this quest, you'll work with a reduced dataset of 6,000 images and use a quantum kernel to predict the digits.

## Historical Context & Challenge Overview

The MNIST (Modified National Institute of Standards and Technology) dataset was introduced by Yann LeCun et al. in 1994 and has served as a fundamental benchmark in the machine learning community for almost 30 years. This collection of handwritten digits has been instrumental in testing and validating numerous computer vision approaches, from traditional machine learning to deep neural networks.

While modern classical methods have achieved near-perfect accuracy on MNIST, our challenge takes a different approach. We're revisiting this iconic benchmark through the lens of quantum machine learning, not with the goal of surpassing classical accuracy records, but to explore novel quantum techniques and methodologies. To make the challenge more suitable for quantum processing, we're working with a reduced dataset of 6,000 images instead of the original 70,000, adding an interesting constraint that makes the problem more challenging and relevant for quantum approaches.

## Photonic Quantum Computing & Perceval

This challenge leverages photonic quantum computing, a promising quantum computing paradigm that uses light particles (photons) as quantum bits. Participants will use the Perceval framework, an open-source platform developed by Quandela for programming photonic quantum computers. You can learn more about Perceval and its capabilities at [perceval.quandela.net](https://perceval.quandela.net).

## Quantum Computing Resources

Participants can develop small scale algorithms using local simulation and in phase 2 will have access to [Scaleway's Quantum-as-a-Service platform](https://labs.scaleway.com/en/qaas/), which provides both large-scale quantum simulators and actual QPU access. This platform enables participants to test and run their quantum algorithms in both simulated and real quantum environments.

## Organization of the repository
The dataset is located in the `data` folder, containing `train.csv` and `test.csv` files. 
The notebook `MNIST_classification_quantum.ipynb` and its equivalent script, `training.py`, contain the training loop used for model training.

An example code for building quantum embeddings and integrating them into a basic classical model is split in separate scripts: the model is defined in `model.py`, the Boson Sampler in `boson_sampler.py` and some helper functions (dataset class for the reduced dataset, accuracy function...) can be found in `utils.py`. 

## Challenge Rules

Use any classical machine learning model and demonstrate improved performance with a quantum model (see Evaluation Criteria).
Submit your solution as a reproducible Jupyter notebook.
Modify the provided quantum model as needed. It can rely on quantum kernels or other methods. 

## Challenge Structure

The challenge consists of two phases:

### Phase 1 

Participants submit an initial solution relying on the provided quantum model. The top 10 solutions will advance to Phase 2 based on:

- Accuracy improvements potential as defined in the Evaluation Criteria
- Creativity in approach
- Report quality

Selected participants will receive credits for Scaleway GPU simulators to develop extended solutions in Phase 2.

### Phase 2 

Qualified participants will further develop and submit enhanced solutions.

## Prizes

The challenge offers exciting rewards for top performers:

- 1st Place: €2,500
- 2nd Place: €1,500
- 3rd Place: €1,000
- Top 10 Teams: Exclusive Quantum Computing Goodies

Additionally, winners will:
- Present their solution in a Perceval webinar
- Have the opportunity to contribute to a scientific publication in collaboration with Quandela

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

- Registration deadline: November 28th, 2024
- Phase 1 submission deadline: January 13th, 2025
- Phase 2 participant announcement: January 17th, 2025
- Phase 2 submission deadline: March 21st, 2025

Good luck! We look forward to your submissions.
