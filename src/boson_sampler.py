# src/boson_sampler.py
import perceval as pcvl
from perceval.providers import scaleway  # import Scaleway Session class if needed

import torch
import torch.nn as nn
import os
import hashlib
import pickle


from abc import ABC, abstractmethod

class InterferometerBuilder(ABC):
    @abstractmethod
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        """Return a Perceval circuit for m modes given a list of phase parameters."""
        pass

def safe_param(i, params):
    # If i is within bounds, return the parameter; otherwise, default to 0.
    return params[i] if i < len(params) else 0

class TriangularInterferometerBuilder(InterferometerBuilder):
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        if parameters is None:
            # Use symbolic parameters: for a triangular mesh you need m*(m-1) parameters.
            parameters = [pcvl.P(f"phi_{i}") for i in range(m * (m - 1))]
        # Build the circuit using the local variable m and the correct variable name (parameters)
        return pcvl.GenericInterferometer(m, lambda idx: (
            pcvl.BS()
            .add(0, pcvl.PS(safe_param(2 * idx, parameters)))
            .add(0, pcvl.BS())
            .add(0, pcvl.PS(safe_param(2 * idx + 1, parameters)))
        ))


class RectangularInterferometerBuilder(InterferometerBuilder):
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        if parameters is None:
            parameters = [pcvl.P(f"phi_rect_{i}") for i in range(m * (m - 1))]
        # For demonstration, we build two layers sequentially.
        # (Note: In practice, you’d implement the proper rectangular mesh.)
        layer1 = pcvl.GenericInterferometer(m, lambda idx: (
                    pcvl.BS().add(0, pcvl.PS(parameters[2 * idx]))
               ))
        layer2 = pcvl.GenericInterferometer(m, lambda idx: (
                    pcvl.BS().add(0, pcvl.PS(parameters[2 * idx + 1]))
               ))
        # If Perceval supports circuit composition via the >> operator, we can compose:
        return layer1 >> layer2




class BosonSampler:
    def __init__(self, m: int, n: int, postselect: int = None, backend: str = "SLOS", session=None, builder: InterferometerBuilder = None, cache_enabled=False, cache_directory="results/cache"):
        """
        Photonic boson sampler for embedding data.
        :param m: number of modes
        :param n: number of photons (n <= m)
        :param postselect: minimum detected photons for a valid output (defaults to n if None)
        :param backend: local simulation backend name (ignored if session is provided)
        :param session: optional Perceval session for remote execution (Scaleway)
        :param builder: An instance of InterferometerBuilder to build the circuit.
        """
        assert n <= m, "Photons n must be <= modes m"
        self.m = m
        self.n = n
        self.postselect = postselect if postselect is not None else n
        assert self.postselect <= n, "Postselect cannot exceed number of photons"
        self.backend = backend
        self.session = session  # If not None, use remote session for simulation
        # If no builder is given, use the triangular mesh as default.
        if builder is None:
            from boson_sampler import TriangularInterferometerBuilder  # ensure proper import
            self.builder = TriangularInterferometerBuilder()
        else:
            self.builder = builder

        # Caching settings
        self.cache_enabled = cache_enabled
        self.cache_directory = cache_directory
        if self.cache_enabled and not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)


    def _compute_hash(self, data_tensor):
        # Compute a unique hash based on the tensor contents.
        # Ensure the tensor is on CPU and convert to a numpy byte string.
        return hashlib.md5(data_tensor.cpu().numpy().tobytes()).hexdigest()


    @property
    def nb_parameters(self) -> int:
        """Number of phase parameters available (used for embedding)."""
        return self.m * (self.m - 1) - (self.m // 2)


    @property
    def embedding_size(self) -> int:
        """Number of output features (possible output states after postselection)."""
        from math import comb
        size = 0
        for k in range(self.postselect, self.n + 1):
            size += comb(self.m, k)
        return size


    def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
        """Delegates the creation of the circuit to the builder."""
        return self.builder.create_circuit(self.m, parameters)

    # TODO... DELTE    
    # def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
    #     """
    #     Create an interferometer circuit with given phase parameters.
    #     If no parameters are provided, uses symbolic placeholders.
    #     """
    #     # (Use Perceval's GenericInterferometer for a mesh of beamsplitters and phase shifters)
    #     if parameters is None:
    #         # Use symbolic parameters if none provided
    #         params = [pcvl.P(f"phi_{i}") for i in range(self.m * (self.m - 1))]
    #         # Duplicate each parameter for paired phase shifters
    #         #params = [val for p in parameters for val in (p, p)]
    #     else:
    #         params = parameters
    #     return pcvl.GenericInterferometer(self.m, lambda idx: (
    #                 pcvl.BS()  # 50:50 beam splitter
    #                 .add(0, pcvl.PS(params[2*idx]))   # phase shifter, then another BS
    #                 .add(0, pcvl.BS())
    #                 .add(0, pcvl.PS(params[2*idx+1]))
    #            ))

    def prepare_processor(self, processor: pcvl.Processor, parameters: list[float]):
        """Configure the given processor with the circuit, input state, and detection filters."""
        processor.set_circuit(self.create_circuit(parameters))
        processor.min_detected_photons_filter(self.postselect)  # enforce minimum detected photons
        processor.thresholded_output(True)  # use threshold detectors (no photon-number resolution)
        # Prepare an input state with n photons evenly spaced in m modes (e.g., [1,0,1,0,...] for 2 photons)
        input_state = [0] * self.m
        if self.n > 0:
            step = max(1, self.m // self.n)
            #step = self.m // self.n
            for i in range(self.n):
                input_state[i * step] = 1
        processor.with_input(pcvl.BasicState(input_state))

    def run(self, parameters: list[float], n_samples: int):
        """
        Run the boson sampler circuit with given phase parameters, sample n_samples times.
        Returns a Perceval BSDistribution (a dict-like object of outcome probabilities).
        """
        if self.session is not None:
            # Build a remote processor via the session (e.g., Scaleway QPU or simulator)
            proc = self.session.build_remote_processor()            # Remote QPU simulation
        else:
            proc = pcvl.Processor(self.backend, self.m)             # Local simulation backend
        # Configure the processor with the circuit and input state
        self.prepare_processor(proc, parameters)
        # Use Perceval's sampling algorithm (Sampler) to run the circuit
        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=n_samples)
        job = sampler.probs(n_samples)  # run sampling
        # If job has the method get_results(), use it; otherwise, assume job is already the result.
        if hasattr(job, "get_results"):
            result = job.get_results() # or job.execute_sync() depending on your workflow
        else:
            result = job

        return result  # BSDistribution of outcomes


    def _distribution_to_feature_vector(self, distribution):
        """
        Convert a distribution dictionary into a fixed-length feature vector.
        """
        feature_vec = torch.zeros(self.embedding_size)
        state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
        for i, state in enumerate(state_list):
            feature_vec[i] = distribution.get(state, 0.0)
        return feature_vec

    def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
        """
        Compute the quantum embedding adaptively.
        Starts with min_samples and doubles until the feature vector converges
        (norm difference < tol) or until max_samples is reached.
        """
        if max_samples is None:
            max_samples = self.n_samples  # default maximum samples from config
        
        prev_embedding = None
        samples = min_samples
        while samples <= max_samples:
            if hasattr(self, "params"):
                distribution = self.run(n_samples=samples)
            else:
                distribution = self.run(parameters=[], n_samples=samples)
            current_embedding = self._distribution_to_feature_vector(distribution)
            if prev_embedding is not None:
                diff = torch.norm(current_embedding - prev_embedding).item()
                if diff < tol:
                    print(f"Adaptive sampling converged with {samples} samples.")
                    return current_embedding
            prev_embedding = current_embedding
            samples *= 2
        print(f"Adaptive sampling reached max samples: {max_samples}")
        return prev_embedding

    # def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
    #     """
    #     Compute the quantum embedding adaptively.
    #     It computes a feature vector (from the output distribution) and doubles the sample count 
    #     until the difference between successive feature vectors is below tol or until max_samples is reached.
    #     """
    #     if max_samples is None:
    #         max_samples = self.n_samples  # default maximum samples from config

    #     # Prepare phase parameters (same as in embed()):
    #     flat = data_tensor.flatten()
    #     if flat.shape[0] > self.nb_parameters:
    #         raise ValueError("Input tensor too large for the number of modes/photons")
    #     phases = torch.zeros(self.nb_parameters)
    #     phases[:flat.shape[0]] = flat
    #     phase_list = (phases * 2 * torch.pi).tolist()

    #     prev_feature = None
    #     samples = min_samples
    #     while samples <= max_samples:
    #         # Run the circuit with the computed phase_list.
    #         distribution = self.run(phase_list, n_samples=samples)
    #         # Convert distribution to a feature vector:
    #         feature_vec = torch.zeros(self.embedding_size)
    #         state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
    #         for i, state in enumerate(state_list):
    #             feature_vec[i] = distribution.get(state, 0.0)
    #         # Compare with the previous feature vector.
    #         if prev_feature is not None:
    #             diff = torch.norm(feature_vec - prev_feature).item()
    #             print(f"Adaptive sampling: {samples} samples, diff = {diff:.6f}")
    #             if diff < tol:
    #                 print(f"Adaptive sampling converged with {samples} samples.")
    #                 return feature_vec
    #         prev_feature = feature_vec
    #         samples *= 2
    #     print(f"Adaptive sampling reached max samples: {max_samples}")
    #     return prev_feature


    # TODO delete
    # def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
    #     """
    #     Compute the quantum embedding adaptively.
    #     Starts with min_samples and doubles until the embedding converges 
    #     (difference less than tol) or until max_samples is reached.
    #     """
    #     if max_samples is None:
    #         max_samples = self.n_samples  # default maximum samples from config
        
    #     prev_embedding = None
    #     samples = min_samples
    #     while samples <= max_samples:
    #         current_embedding = self.run(parameters=[], n_samples=samples)
    #         if prev_embedding is not None:
    #             diff = torch.norm(current_embedding - prev_embedding).item()
    #             if diff < tol:
    #                 print(f"Adaptive sampling converged with {samples} samples.")
    #                 return current_embedding
    #         prev_embedding = current_embedding
    #         samples *= 2
    #     print(f"Adaptive sampling reached max samples: {max_samples}")
    #     return prev_embedding



    def embed(self, data_tensor, n_samples: int):
        """
        Embed an input tensor (image data) into a quantum feature vector.
        The input tensor values (assumed in [0,1]) are mapped to phase shifts.
        :param data_tensor: input torch tensor with values in [0,1]
        :param n_samples: number of samples for Monte Carlo estimation (ignored if using an analytic mode)
        :return: torch tensor of shape (embedding_size,) with estimated output probabilities.
        """

        # If caching is enabled, try to load the cached embedding.
        if getattr(self, "cache_enabled", False):
            key = hashlib.md5(data_tensor.cpu().numpy().tobytes()).hexdigest()
            cache_file = os.path.join(self.cache_directory, f"{key}.pkl")
            if os.path.exists(cache_file):
                print("Loading embedding from cache.")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        # Compute embedding if not cached.
        flat = data_tensor.flatten()
        if flat.shape[0] > self.nb_parameters:
            raise ValueError("Input tensor too large for the number of modes/photons")


        # Prepare phase parameters: pad/truncate to fill the interferometer.
        # Pad or truncate the phases vector to fill all required phase shifters
        #self.nb_parameters == phase_count = self.m * (self.m - 1)  # number of phase parameters needed for full interferometer
        phases = torch.zeros(self.nb_parameters)
        # Use data values (scaled 0-1) for the first part of phases, remaining stay 0
        phases[:flat.shape[0]] = flat
        # Scale phases from [0,1] to [0, 2π] as phase shifts
        phase_list = (phases * 2 * torch.pi).tolist()


        # For variational sampler, self.run expects only n_samples.
        if hasattr(self, "params"):
            distribution = self.run(n_samples=n_samples)
        else:
            distribution = self.run(parameters=phase_list, n_samples=n_samples)
            
        ## Run the circuit and get the output distribution
        ##distribution = self.run(phase_list, n_samples)


        # Convert distribution to a fixed-length probability vector
        feature_vec = torch.zeros(self.embedding_size)
        state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
        for i, state in enumerate(state_list):
            feature_vec[i] = distribution.get(state, 0.0)


        # Save the computed embedding to cache.
        if getattr(self, "cache_enabled", False):
            with open(cache_file, "wb") as f:
                pickle.dump(feature_vec, f)


        return feature_vec

    #@lru_cache(maxsize=1)
    def _generate_all_output_states(self):
        """Precompute all possible output basis states (with postselection).
        Generate and cache all possible output states (as BasicState objects).
        """
        states = []
        for k in range(self.postselect, self.n + 1):
            from itertools import combinations
            for ones in combinations(range(self.m), k):
                bitstring = [1 if j in ones else 0 for j in range(self.m)]
                states.append(pcvl.BasicState(bitstring))
        return states

    from concurrent.futures import ThreadPoolExecutor

    @staticmethod
    def parallel_embed(self, image_list, n_samples, adaptive=False, min_samples=100, tol=1e-3):
        """
        Compute embeddings for a list of images concurrently.
        Parameters:
          image_list : List of image tensors.
          n_samples  : Maximum number of samples for each embedding.
          adaptive   : If True, uses adaptive_embed; otherwise uses embed.
          min_samples: Starting sample count for adaptive sampling.
          tol        : Tolerance for convergence in adaptive sampling.
        Returns:
          A list of embeddings.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for img in image_list:
                if adaptive:
                    futures.append(executor.submit(self.adaptive_embed, img, min_samples, n_samples, tol))
                else:
                    futures.append(executor.submit(self.embed, img, n_samples))
            embeddings = [f.result() for f in futures]
        return embeddings






class VariationalBosonSampler(BosonSampler):
    def __init__(self, m: int, n: int, postselect: int = None, backend: str = "SLOS", 
                 session=None, builder: InterferometerBuilder = None, cache_enabled=False, 
                 cache_directory="results/cache", init_params=None):
        """
        A variant of BosonSampler where the phase parameters are trainable.
        """
        super().__init__(m, n, postselect, backend, session, builder, cache_enabled, cache_directory)
        # Initialize trainable parameters. If not provided, initialize to small constant values.
        if init_params is None:
            init_params = [0.1] * self.nb_parameters
        # Register parameters as a torch.nn.Parameter so they are updated during training.
        self.params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))

    adaptive_embed = BosonSampler.adaptive_embed


    def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
        """
        Overrides the create_circuit method to use trainable parameters.
        """
        # Use the current trainable parameters from self.params
        phase_list = self.params.tolist()
        return self.builder.create_circuit(self.m, phase_list)
    
    def run(self, n_samples: int, parameters: list[float] = None):
        """
        Overrides run() to build the circuit using trainable parameters.
        The 'parameters' argument is ignored since self.params is used.
        """
        circuit = self.create_circuit()  # This uses self.params internally.
        if self.session is not None:
            proc = self.session.build_remote_processor()
        else:
            proc = pcvl.Processor(self.backend, self.m)
        # We ignore the passed 'parameters' and use self.params instead.
        self.prepare_processor(proc, parameters=[])  
        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=n_samples)
        job = sampler.probs(n_samples)
        if hasattr(job, "get_results"):
            result = job.get_results()
        else:
            result = job
        return result




##############
# Enhanced BosonSampler with Caching
# Implementing caching to avoid redundant quantum computations:
############
from functools import lru_cache
import os
import pickle
import hashlib

class CachedBosonSampler(BosonSampler):
    """
    Extended BosonSampler with caching capabilities to avoid redundant computations.
    """
    def __init__(self, m, n, postselect=None, session=None, 
                 cache_enabled=True, cache_size=1000, 
                 disk_cache=False, cache_dir=None):
        super().__init__(m, n, postselect, session)
        self.cache_enabled = cache_enabled
        self.disk_cache = disk_cache
        self.cache_dir = cache_dir
        
        if disk_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Use the wrapper function to apply caching
        if cache_enabled:
            self.embed = self._cached_embed(self.embed, maxsize=cache_size)
    
    def _cached_embed(self, func, maxsize=1000):
        """Apply in-memory caching with LRU strategy."""
        memory_cache = lru_cache(maxsize=maxsize)(func)
        
        def wrapped_func(t, n_sample):
            # Convert tensor to tuple for hashing
            t_tuple = tuple(t.reshape(-1).tolist())
            
            # Check disk cache if enabled
            if self.disk_cache and self.cache_dir:
                cache_key = hashlib.md5(str(t_tuple).encode()).hexdigest()
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Use in-memory cache or compute
            result = memory_cache(t, n_sample)
            
            # Save to disk cache if enabled
            if self.disk_cache and self.cache_dir:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                    
            return result
        
        return wrapped_func if self.cache_enabled else func
