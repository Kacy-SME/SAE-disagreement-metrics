"""
Theoretical grounding for the SAE concept-alignment conjecture.

This module documents conjectures, supporting lemmas, and citation-backed
claims for the paper. It contains no executable logic—only dataclasses and
instantiated theory objects for reference and reproducible prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

LemmaStatus = Literal["proven", "empirically_supported", "open"]


@dataclass(frozen=True)
class SupportingLemma:
    """A supporting lemma in the theoretical chain for SAE concept alignment."""

    id: str
    title: str
    status: LemmaStatus
    statement: str
    citations: List[str]
    notes: str = ""


@dataclass(frozen=True)
class ConjunctureStatus:
    """
    High-level status of a conjecture: dependencies, evidence, and open questions.

    Note: named ``ConjunctureStatus`` per paper module spec (conjecture lifecycle).
    """

    name: str
    statement: str
    depends_on: List[str]
    empirical_evidence: str
    open_questions: List[str]


# ---------------------------------------------------------------------------
# Main conjecture
# ---------------------------------------------------------------------------

SAE_CONCEPT_ALIGNMENT_CONJECTURE = """
**Conjecture (SAE Concept Alignment).**

Let φ: 𝒳 → ℝᵈ be a ViT encoder trained with a DINO-style local–global
consistency objective, and let ℱ = {f₁, …, fₖ} be the SAE latents learned by
minimizing reconstruction loss with sparsity constraint ‖h‖₀ ≤ L₀ over patch
activations φₗ(x) at layer ℓ.

Let 𝒞 = {c₁, …, cₘ} be a set of expert-defined domain concepts (e.g. HiRISE
landmark classes) with associated probe directions {pⱼ}ⱼ₌₁ᵐ learned by linear
classifiers on φₗ.

Then for sufficiently disentangled φ (absorption proxy < τ) and sufficiently
large dictionary k ≫ d, there exists a subset ℱ* ⊆ ℱ such that for each
cⱼ ∈ 𝒞, at least one fᵢ ∈ ℱ* satisfies cos(dᵢ, pⱼ) > δ for decoder direction
dᵢ and threshold δ > 0.
""".strip()


# ---------------------------------------------------------------------------
# Supporting lemmas
# ---------------------------------------------------------------------------

LEMMA_1_SIZE_PRINCIPLE = SupportingLemma(
    id="lemma_1",
    title="Size Principle",
    status="proven",
    statement="""
**Lemma 1 — Size Principle.**

If concept examples {x⁽ⁿ⁾} are sampled i.i.d. uniformly from concept extension
C ⊆ 𝒳, then under the Bayesian framework the posterior probability of hypothesis
h satisfies

    p(h | X) ∝ p(h) · |h|⁻ⁿ,

making smaller consistent hypotheses exponentially more probable as n increases.
""".strip(),
    citations=[
        'Tenenbaum (1998), "Bayesian Modeling of Human Concept Learning," '
        "MIT PhD thesis.",
        'Tenenbaum & Griffiths (2001), "Generalization, similarity, and Bayesian '
        'inference," Behavioral and Brain Sciences.',
    ],
    notes=(
        "Provides a normative bias toward compact hypotheses; motivates why "
        "sparse, low-cardinality explanations (SAE latents) are preferred when "
        "consistent with data."
    ),
)

LEMMA_2_IB_COMPRESSION_BOUND = SupportingLemma(
    id="lemma_2",
    title="IB Compression Bound",
    status="proven",
    statement="""
**Lemma 2 — IB Compression Bound.**

For any encoder Z = φ(X) and task labels Y, the Information Bottleneck tradeoff

    min I(Z; X) − β I(Z; Y)

implies that task-optimized encoders maximize I(Z; Y) at the cost of compressing
I(Z; X) toward the task-relevant subspace, leaving cross-task variation entangled
rather than factored.

Multi-task objectives (e.g. EVL merge across sensors) that optimize for multiple
Yᵢ simultaneously produce representations where no single sparse basis aligns
with any individual Yᵢ.
""".strip(),
    citations=[
        'Tishby et al. (1999), "The information bottleneck method," '
        "37th Allerton Conference.",
        'Sanneman & Shah (2024), "An Information Bottleneck Characterization of '
        'the Understanding-Workload Tradeoff in Human-Centered Explainability," '
        "ACM TOCHI (empirical result: distortion I(Z; X) − I(Z; Y) negatively "
        "correlates with feature understanding at ρ = −0.83, p < 0.001).",
    ],
    notes=(
        "Explains why general-purpose orbital encoders may entangle landform "
        "signal with sensor- and scene-level nuisance factors."
    ),
)

LEMMA_3_SUPERPOSITION_RESOLUTION = SupportingLemma(
    id="lemma_3",
    title="Superposition Resolution",
    status="empirically_supported",
    statement="""
**Lemma 3 — Superposition Resolution.**

In a representation space with absorption proxy < τ, SAE training with dictionary
size k ≫ d and sparsity L₀ ≪ k recovers decoder directions that are
approximately orthogonal to each other and align with the principal directions of
variation in the activation distribution.

Under the linear representation hypothesis, if concept cⱼ activates a consistent
direction in φₗ, then a decoder column dᵢ will align with that direction when the
absorption proxy is low.
""".strip(),
    citations=[
        'Elhage et al. (2022), "Toy Models of Superposition," '
        "Transformer Circuits Thread.",
        'Bricken et al. (2023), "Towards Monosemanticity," '
        "Transformer Circuits Thread.",
        'Karvonen et al. (2025), "SAEBench," ICML (absorption and SCR metrics '
        "show substantial architectural differences; hierarchical architectures "
        "outperform by 30–40%).",
    ],
    notes=(
        "Empirically validated in this project via layer-depth absorption trends "
        "(e.g. mars_orbital_vit early ≈ 0.998 → late ≈ 0.73) and SAEBench-style "
        "core metrics."
    ),
)

LEMMA_4_PROBE_SAE_DIRECTION_CORRESPONDENCE = SupportingLemma(
    id="lemma_4",
    title="Probe–SAE Direction Correspondence",
    status="open",
    statement="""
**Lemma 4 — Probe–SAE Direction Correspondence.**

If a linear probe pⱼ achieves F1 > ε on concept cⱼ in activation space φₗ,
then there exists a decoder direction dᵢ in the SAE trained on φₗ such that
cos(dᵢ, pⱼ) > δ.

This is the open step in the theoretical chain. Existence of the probe guarantees
the concept direction exists in the activation space; it does not guarantee the
SAE training procedure finds it rather than other statistically prominent
directions.

**Empirical validation criterion:** After class-balanced probing on HiRISE v3.2
landmark classes (dropping class 0 "other"), if SAE latent F1 substantially
exceeds the majority-class baseline (~0.125 macro-F1) on named landform classes
using the mars_orbital_vit late-layer SAE, this lemma is empirically supported
for the domain.
""".strip(),
    citations=[
        (
            "Colin et al. (2022), What I Cannot Predict, I Do Not Understand: "
            "A Human-Centered Evaluation Framework for Explainability Methods, "
            "NeurIPS (faithfulness–utility disconnect: probe existence does not "
            "guarantee SAE alignment)."
        ),
        (
            "Gurnee et al. (2023), Finding Neurons in a Haystack, COLM "
            "(k-sparse probing methodology used in SAEBench sparse probing evaluation)."
        ),
    ],
    notes=(
        "Target of empirical closure in this paper via --probe-landforms-only "
        "sparse probing (7 named classes, n ≈ 87 eval images)."
    ),
)

SUPPORTING_LEMMAS: List[SupportingLemma] = [
    LEMMA_1_SIZE_PRINCIPLE,
    LEMMA_2_IB_COMPRESSION_BOUND,
    LEMMA_3_SUPERPOSITION_RESOLUTION,
    LEMMA_4_PROBE_SAE_DIRECTION_CORRESPONDENCE,
]


# ---------------------------------------------------------------------------
# Conjecture status (paper-facing summary)
# ---------------------------------------------------------------------------

CONJECTURE_STATUS = ConjunctureStatus(
    name="SAE Concept Alignment",
    statement=SAE_CONCEPT_ALIGNMENT_CONJECTURE,
    depends_on=[
        LEMMA_1_SIZE_PRINCIPLE.title,
        LEMMA_2_IB_COMPRESSION_BOUND.title,
        LEMMA_3_SUPERPOSITION_RESOLUTION.title,
        LEMMA_4_PROBE_SAE_DIRECTION_CORRESPONDENCE.title,
    ],
    empirical_evidence=(
        "Absorption proxy layer-depth trend: mars_orbital_vit Δ = −0.121 "
        "(early → late disentanglement) vs MOMO Δ = +0.003 across backbone "
        "comparison (n = 6 orbital backbones). Landform-balanced sparse probing "
        "(HiRISE v3.2 official test split, classes 1–7): report sparse-probe macro-F1 "
        "on n = 311 creator-held-out originals with probe trained on official train; "
        "use sparse_probe_f1_*_core_classes for the five test classes with n ≥ 20. "
        "Prior ad-hoc holdout (n ≈ 87) exceeded the ~0.125 majority-class baseline—supporting Lemma 4 for "
        "this domain when read together with low late-layer absorption (~0.73). "
        "Paper contribution: empirical closure of Lemma 4 via the landform-balanced "
        "probe experiment (--probe-landforms-only)."
    ),
    open_questions=[
        "Lemma 4 (Probe–SAE Direction Correspondence): general proof remains open; "
        "probe existence in φₗ does not imply the SAE optimization recovers aligned "
        "decoder columns rather than dominant but task-irrelevant directions.",
        "Per-class F1 on rare HiRISE landforms (e.g. impact ejecta, spider) is "
        "often zero at n ≈ 87; stronger claims require more labeled data or "
        "explicit cos(dᵢ, pⱼ) measurement.",
        "Cross-backbone 'late' layers are fractional depths (3L/4), not identical "
        "absolute indices across ViT-Base (12L) vs ViT-Large (24L).",
    ],
)
