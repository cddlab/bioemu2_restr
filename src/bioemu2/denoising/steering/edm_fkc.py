import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import torch

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.context_parallel.models.score_model import CPTrunkEmbeddings
from bioemu2.denoising.augmentation import (
    augment_coordinates,
    compute_random_augmentation,
)
from bioemu2.denoising.edm import (
    edm_sample_schedule,
)
from bioemu2.denoising.steering.conditioning import resample_conditioning
from bioemu2.denoising.steering.feats import (
    Tracker,
    grouped_softmax,
    systematic_resample,
)
from bioemu2.denoising.steering.potentials import Potential
from bioemu2.model import InferenceModel
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.inference_inputs import broadcast_multiplicity

Reheat = Literal["sigma", "gamma"]
WeightUpdate = Literal["smc", "guidance"]


def _karras_sigma(t: float, *, sigma_min: float, sigma_max: float, sigma_data: float, rho: int):
    """Analytic Karras sigma schedule for data-to-noise time ``t``."""
    sigma_min_root = sigma_min ** (1.0 / rho)
    sigma_max_root = sigma_max ** (1.0 / rho)
    return sigma_data * (sigma_min_root + t * (sigma_max_root - sigma_min_root)) ** rho


def _normalized_time_from_sigma(
    sigma: float, *, sigma_min: float, sigma_max: float, sigma_data: float, rho: int
) -> float:
    """Invert the Karras sigma schedule to its noisy-to-clean normalized time.

    Churn can push ``sigma_hat`` off the grid, so times are read back from sigma
    rather than from the step index. The result is clamped to ``[0, 1]``: beyond
    that range the anneal would invert or overshoot the tilt.
    """
    sigma_normalized = max(sigma / sigma_data, sigma_min)
    sigma_root = sigma_normalized ** (1.0 / rho)
    sigma_min_root = sigma_min ** (1.0 / rho)
    sigma_max_root = sigma_max ** (1.0 / rho)
    fraction = (sigma_root - sigma_max_root) / (sigma_min_root - sigma_max_root)
    return min(max(1.0 - fraction, 0.0), 1.0)


def edm_fkc_sampler(
    *,
    batch: BatchAllAtomStructure,
    num_steps: int,
    score_model: InferenceModel[TrunkEmbeddings | CPTrunkEmbeddings],
    trunk_embeddings: TrunkEmbeddings,
    rho: int = 7,
    gamma_0: float = 0.8,
    gamma_min: float = 0.0,
    multiplicity: int = 1,
    potentials: Sequence[Potential] = (),
    resampling_interval: int | None = None,
    resampling_group_size: int | None = None,
    ess_threshold: float | None = 0.9,
    weight_update: WeightUpdate = "smc",
    beta_power: float = 4.0,
    reheat: Reheat = "gamma",
    tracker: Tracker | None = None,
) -> BatchAllAtomStructure:
    r"""Sample with EDM transport, a terminal zero step, and denoised potentials.

    See ``docs/steering.md`` for the target distribution, reheat modes, and
    weight constructions.

    Args:
        batch: Structure batch defining the system to sample. Its coordinates are
            broadcast to ``multiplicity`` before sampling; the current batch
            is the resampling population.
        num_steps: Number of positive levels on the Karras grid. The schedule adds
            a terminal ``sigma=0``, so the loop runs ``num_steps`` transitions.
        score_model: Denoiser returning ``denoised_coords`` for given coordinates
            and noise levels.
        trunk_embeddings: Reusable sequence-trunk outputs. Static context is
            prepared once for this sampler invocation.
        rho: Karras grid exponent controlling step spacing.
        gamma_0: Multiplicative churn factor, used only by ``reheat="gamma"``.
        gamma_min: Threshold below which gamma reheating is
            switched off, matching the AF3 gate. (1 + gamma_0) sigma_i |_{sigma_i < gamma_min} = sigma_i
        multiplicity: Number of conformations to sample for the single input
            structure. They are also the FKC particle population, resampled within
            independent groups of ``resampling_group_size`` (all of them when ``None``).
        potentials: Objectives evaluated on denoised coordinates. FKC is active
            only when this is non-empty.
        resampling_interval: Steps between systematic resamples. Mutually
            exclusive with ``ess_threshold``.
        resampling_group_size: Particles per independent resampling group;
            ``None`` resamples the whole batch as one group.
        ess_threshold: Effective-sample-size fraction that triggers a resample.
            Mutually exclusive with ``resampling_interval``.
        weight_update: Importance-weight construction, ``"smc"`` or ``"guidance"``.
        beta_power: Exponent of the annealing numerator ``(1 - t)^p``; see
            ``docs/steering.md`` for the noise-dependent normalization.
        reheat: Noise level each transition restarts from. ``"gamma"`` applies
            AF3-gated churn :math:`\hat\sigma_i = (1 + \gamma_0)\sigma_i` while
            :math:`\sigma_{i+1} > \gamma_{\min}`; ``"sigma"`` reheats to the
            previous scheduled level, leaving the first transition unreheated.
        tracker: Optional Tracker for saving ``x_t``, denoised ``x0``, per-potential
            energies, weights, sigma, and beta at the end of each step, before
            resampling. Coordinates are in angstroms; energies are evaluated on
            ``x0`` and used for weighting in SMC, but only saved as diagnostics
            in guidance mode. Outside SMC, tracking adds one denoiser call per
            nonterminal step. Pass ``None`` to disable tracking.

    Returns:
        Sampled structures.

    Raises:
        ValueError: If a potential energy or particle log-weight is non-finite,
            if ``reheat`` is unknown, if ``resampling_interval`` and
            ``ess_threshold`` are both or neither set, if either trigger or
            ``resampling_group_size`` is out of range, if ``weight_update`` is unknown, if
            ``weight_update`` is ``"guidance"`` while ``reheat`` is not
            ``"sigma"``.
    """
    if getattr(batch, "restraints_config", None) is not None:
        raise NotImplementedError(
            "RGI supports edm, edm_gd and edm_fk; edm_fkc requires an "
            "RGI-aware importance-weight derivation."
        )
    potentials = tuple(potentials)
    if reheat not in ("sigma", "gamma"):
        raise ValueError(f"edm_fkc_sampler got unknown reheat={reheat!r}")
    if resampling_interval is not None and ess_threshold is not None:
        raise ValueError(
            "edm_fkc_sampler accepts at most one resampling trigger, got "
            f"resampling_interval={resampling_interval!r} and ess_threshold={ess_threshold!r}"
        )
    if resampling_interval is None and ess_threshold is None:
        raise ValueError(
            "edm_fkc_sampler needs exactly one of resampling_interval and ess_threshold, "
            "got neither"
        )
    if resampling_interval is not None and resampling_interval < 1:
        raise ValueError(f"edm_fkc_sampler got resampling_interval={resampling_interval} < 1")
    if resampling_group_size is not None and resampling_group_size < 2:
        raise ValueError(
            f"edm_fkc_sampler got resampling_group_size={resampling_group_size}; use at "
            "least 2, or resampling_interval=None to disable resampling"
        )
    if ess_threshold is not None and not 0.0 <= ess_threshold <= 1.0:
        raise ValueError(f"edm_fkc_sampler got ess_threshold={ess_threshold} outside [0, 1]")
    if weight_update not in ("smc", "guidance"):
        raise ValueError(f"edm_fkc_sampler got unknown weight_update={weight_update!r}")
    if weight_update == "guidance" and reheat != "sigma":
        raise ValueError("edm_fkc_sampler requires reheat='sigma' for guidance weights")

    context = score_model.compute_static_context(batch, trunk_embeddings, multiplicity)
    is_ito = context.is_dynamics_conditioned is not None and bool(
        context.is_dynamics_conditioned.any()
    )
    fkc_active = bool(potentials)

    device = batch.coords.device
    schedule = score_model.edm_schedule
    sigma_min = schedule.sigma_min
    sigma_max = schedule.sigma_max
    sigma_data = schedule.sigma_data
    # Read the grid back to floats once: the loop branches on sigma every step.
    sigmas = edm_sample_schedule(
        num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
        device=device,
    ).tolist()
    num_transitions = len(sigmas) - 1

    if batch.coords.shape[0] != 1:
        # Resampling across several conditions (structures or ITO starting
        # frames) is undefined; the per-condition population is the
        # multiplicity axis.
        raise NotImplementedError("Steered samplers take one structure per call.")
    batch = batch.replace(coords=broadcast_multiplicity(batch.coords, multiplicity))
    atom_mask = batch.atom_mask.repeat_interleave(multiplicity, 0)
    if tracker is not None:
        for index, potential in enumerate(potentials):
            tracker.register(
                f"{index}_{type(potential).__name__}",
                (num_transitions, multiplicity),
            )
        tracker.register("x_t", (num_transitions, *batch.coords.shape))
        tracker.register("x0", (num_transitions, *batch.coords.shape))
        tracker.register("weights", (num_transitions, multiplicity))
        tracker.register("sigma", (num_transitions,))
        tracker.register("beta", (num_transitions,))

    def denoise(coords: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """Evaluate ``Denoiser(coords, sigma)``, keeping the graph if the caller has one.

        A tensor ``sigma`` is expanded rather than rebuilt, so a differentiable
        sigma-leaf survives into the denoiser and ``dU/dsigma`` can be read off.
        """
        sigmas = (
            sigma.expand(coords.shape[0])
            if isinstance(sigma, torch.Tensor)
            else torch.full((coords.shape[0],), sigma, device=device)
        )
        return score_model.denoise(
            context=context,
            trunk_embeddings=trunk_embeddings,
            coords=coords,
            sigmas=sigmas,
        ).denoised_coords

    def potential_energy(
        denoised: torch.Tensor, t_norm: float
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sum every steering potential over denoised coordinates.

        Args:
            denoised: Denoised coordinates, shape ``[particles, atoms, 3]``,
                including the topology's padded atoms.
            t_norm: Normalized time at which each potential resolves its schedule.

        Returns:
            Total energy, shape ``[particles]``, and a list of per-potential
            energies in potential order, each with shape ``[particles]``.
        """
        total = torch.zeros(denoised.shape[0], device=device, dtype=torch.float32)
        energies = []
        for potential in potentials:
            value = potential(denoised.float(), t=t_norm).float()
            energies.append(value)
            total = total + value
        if energies:
            # Transfer flags together, avoiding a host synchronization per potential.
            finite = torch.stack(
                [torch.isfinite(value).all() for value in [*energies, total]]
            ).tolist()
            for index, valid in enumerate(finite[:-1]):
                if not valid:
                    raise ValueError(
                        f"Non-finite energy from potential {index} "
                        f"({type(potentials[index]).__name__}) at t={t_norm}"
                    )
            if not finite[-1]:
                raise ValueError(f"Non-finite total potential energy at t={t_norm}")
        return total, energies

    def beta_for_sigma(sigma: float) -> torch.Tensor:
        """Return beta for the annealed tilt ``exp(-beta * U)``.

        The numerator ``(1 - t)**beta_power`` starts steering at zero strength
        and increases it as denoising proceeds. Larger powers delay steering.
        The endpoint normalizer makes beta reach 1 at the clean endpoint.

        The denominator ``1 + 0.1 * sigma**2`` limits high-noise guidance:
        the guidance coefficient is ``reheat_variance * beta / 2``, and that
        variance grows with sigma squared. Use absolute sigma, including
        ``sigma_data``, so the denominator and reheat variance have the same
        noise scale.

        See ``docs/steering.md`` for the schedule and target distribution.
        """
        t_norm = _normalized_time_from_sigma(
            sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
        )
        sigma_absolute = _karras_sigma(
            t_norm,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
        )
        denom_scale = 0.1
        endpoint_normalizer = 1.0 + denom_scale * (sigma_data * sigma_min) ** 2
        beta = (
            endpoint_normalizer
            * (1.0 - t_norm) ** beta_power
            / (1.0 + denom_scale * sigma_absolute**2)
        )
        return torch.tensor(beta, device=device, dtype=torch.float32)

    atom_coords = sigmas[0] * torch.randn_like(batch.coords)
    log_weight = torch.zeros(atom_coords.shape[0], device=device, dtype=torch.float32)
    # Previous annealed log-potential, -beta(t) U(D(x, sigma), t), carried into
    # the next reheat leg. It starts at zero because beta(t=1) vanishes.
    potential_current = torch.zeros_like(log_weight)
    last_step = num_transitions - 1

    for step_index, (sigma_current, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        # step_index: 0=Noise -> -1=Data; sigmas decrease to 0; sigma_next is 0 on the last step.
        rotation, translation = compute_random_augmentation(
            atom_coords.shape[0], device=device, dtype=atom_coords.dtype
        )  # translation is N(0,1) in each dimension, rotation is a random 3D rotation matrix
        atom_coords = augment_coordinates(atom_coords, rotation, translation, atom_mask)

        if is_ito:
            assert context.coords_cond is not None
            context = replace(
                context,
                coords_cond=augment_coordinates(
                    context.coords_cond, rotation, translation, atom_mask
                ),
            )
        # Reheating is either done to the previous grid level, or with a fixed multiplicative churn
        if reheat == "sigma":
            # The previous grid level: its gap shrinks as the grid is refined, which
            # is what makes the local guidance correction well defined.
            sigma_hat = sigmas[step_index - 1] if step_index > 0 else sigma_current
        else:
            # Fixed multiplicative churn, gated on the destination level as in AF3.
            sigma_hat = sigma_current * (1.0 + (gamma_0 if sigma_next > gamma_min else 0.0))
        reheat_variance = sigma_hat**2 - sigma_current**2
        reheated = atom_coords + math.sqrt(reheat_variance) * torch.randn_like(atom_coords)

        # Annealing times come off the step index, but t_hat is read back from the
        # reheated sigma: churn can push it off-grid, between two scheduled steps.
        t_current = max(1.0 - step_index / num_transitions, 0.0)
        t_next = max(1.0 - (step_index + 1) / num_transitions, 0.0)
        t_hat = _normalized_time_from_sigma(
            sigma_hat,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            rho=rho,
        )
        beta_next = beta_for_sigma(sigma_next)

        guidance_active = fkc_active and weight_update == "guidance"
        smc_active = fkc_active and weight_update == "smc"
        if guidance_active:
            # Compute x0 and the quantities required for guidance
            # grad U is taken w.r.t. the reheated state but *through* the denoiser,
            # so the tilt is defined on Denoiser(x_hat, sigma_hat), not on noisy coords.
            # sigma is a leaf too: U is evaluated on D(x, sigma), so the tilt is
            # explicitly sigma-dependent and its d/dsigma enters the weight below.
            with torch.enable_grad():
                leaf = reheated.detach().requires_grad_(True)
                sigma_leaf = torch.full(
                    (reheated.shape[0],),
                    sigma_hat,
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                denoised = denoise(leaf, sigma_leaf)
                energy_hat, _ = potential_energy(denoised, t_hat)
                gradient, grad_sigma = torch.autograd.grad(
                    energy_hat.sum(),
                    (leaf, sigma_leaf),
                )
            denoised = denoised.detach()
        else:
            with torch.no_grad():
                denoised = denoise(reheated, sigma_hat).detach()

        score = (denoised - reheated) / sigma_hat**2
        if guidance_active:
            beta_hat = beta_for_sigma(sigma_hat)
            beta_current = beta_for_sigma(sigma_current)
            with torch.no_grad():
                energy_current, _ = potential_energy(denoised, t_current)
                energy_next, _ = potential_energy(denoised, t_next)

            score = (denoised - reheated) / sigma_hat**2
            alignment = (
                ((-beta_hat * gradient) * score * (sigma_current**2 - sigma_next**2) / 2.0)
                .flatten(1)
                .sum(dim=1)
            )
            # U depends on sigma through D(x, sigma), so its partial derivative
            # contributes over the base grid step in addition to the discrete
            # beta/potential change.
            log_weight = (
                log_weight
                - beta_next * energy_next
                + beta_current * energy_current
                + alignment
                + beta_hat * grad_sigma * (sigma_current - sigma_next)
            )
            guidance_shift = -0.5 * reheat_variance * beta_hat * gradient
        elif smc_active:
            with torch.no_grad():
                energy_hat, _ = potential_energy(denoised, t_hat)
                potential_hat = -beta_for_sigma(sigma_hat) * energy_hat
            log_weight = log_weight + potential_hat - potential_current

        # Unit step scale: the weights above assume the exact reverse transition.
        atom_coords_next = reheated + sigma_hat * (sigma_hat - sigma_next) * score
        if guidance_active:
            atom_coords_next = atom_coords_next + guidance_shift

        if smc_active or tracker is not None:
            with torch.no_grad():
                denoised_next_coords = (
                    denoise(atom_coords_next, sigma_next).detach()
                    if sigma_next > 0.0
                    else atom_coords_next
                )
                total_next_energy, per_potential_next_energies = potential_energy(
                    denoised_next_coords, t_next
                )
                potential_next = -beta_next * total_next_energy
        if smc_active:
            log_weight = log_weight + potential_next - potential_hat

        # On CUDA, this Python condition waits for the GPU to finish the finite-value check.
        if not torch.isfinite(log_weight).all():
            raise ValueError(f"Non-finite particle log-weight at step {step_index}")
        weight = grouped_softmax(log_weight, resampling_group_size)
        if tracker is not None:
            metrics = {
                f"{index}_{type(potential).__name__}": energy
                for index, (potential, energy) in enumerate(
                    zip(potentials, per_potential_next_energies, strict=True)
                )
            }
            metrics.update(
                {
                    "x_t": atom_coords_next,
                    "x0": denoised_next_coords,
                    "weights": weight,
                    "sigma": torch.tensor(sigma_next, dtype=torch.float32),
                    "beta": beta_next,
                }
            )
            tracker.log(metrics)

        if fkc_active:
            if ess_threshold is not None:
                # Adaptive: resample once any group's effective sample size
                # ratio drops below the threshold, rather than on a fixed cadence.
                ess_ratio = min(
                    1.0 / (len(w) * w.square().sum()).item()
                    for w in weight.split(resampling_group_size or multiplicity)
                )
                do_resample = ess_ratio < ess_threshold
            else:
                assert resampling_interval is not None
                do_resample = (step_index + 1) % resampling_interval == 0
            do_resample = do_resample or step_index == last_step
            if do_resample:
                indices = systematic_resample(weight, resampling_group_size)
                atom_coords_next = atom_coords_next[indices]
                atom_mask = atom_mask[indices]
                if is_ito:
                    context = resample_conditioning(context, indices)
                log_weight = torch.zeros(
                    atom_coords_next.shape[0], device=device, dtype=torch.float32
                )
                if smc_active:
                    potential_next = potential_next[indices]
            if smc_active:
                potential_current = potential_next

        atom_coords = atom_coords_next

    return batch.replace(coords=atom_coords)
