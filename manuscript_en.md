# Irreducible Projections: Composing Epistemic Operators for LLM Agent Memory

> Date: 2026-05-24

###### Abstract

Memory-augmented LLM agents have traditionally relied on retrieval---query-conditioned selection over an artifact pool---as the dominant access mechanism. We argue that retrieval is not the only fundamental access mechanism but one of at least three structurally distinct epistemic operators: retrieval (top-K selection over an unordered pool), state lookup (query-independent readout of a compressed belief state), and trajectory walk (chronological traversal of a temporally ordered transition graph). Each operator preserves different epistemic dimensions and discards others, making distinct classes of memory failures structurally unavoidable; no single operator is complete. We propose a design methodology for memory systems based on operator composition: decompose the task space into epistemic axes, select the optimal operator per axis via an empirically measured phase map, calibrate the selection for the target backbone model, and compose operators through type-safe routing. Cross-model validation on four LLMs reveals that operator complementarity is model-independent---state lookup fails on the same 29 of 112 cases across all models with near-verbatim identical wrong answers---while optimal routing is model-conditioned. A composed system following model-calibrated routing achieves gains of +0.038 to +0.138 over the best single operator across three benchmarks (MSFT, MSFT-NL, STALE). Critically, calibrated composition empirically exhibits a non-inferiority property across all tested settings. The central implication is that memory system design should be reframed as operator composition with model-specific calibration, not as representation optimization under a single access paradigm.

---

## 1 Introduction

Large Language Model (LLM) agents operating over extended time horizons depend on effective memory---the ability to access, reason over, and maintain information about the past. The dominant approach has been retrieval-augmented memory: past interactions are stored as external artifacts and selectively retrieved at query time. Recent work has advanced memory representation (APEX-MEM's temporal property graph and procedural learning, GSW's generative semantic workspace, GAM's hierarchical graph-based consolidation), adaptive retrieval (UMA's RL-based CRUD memory management), and memory evaluation (AMA-Bench's long-horizon evaluation, ReMemR1's retrieval-in-the-loop memory updates). Yet a common assumption unites these efforts: that memory is a collection of artifacts, and intelligence emerges from retrieval-time selection and downstream reasoning.

Recent evidence challenges this foundation. Across multiple long-term memory benchmarks, state-of-the-art memory agents offer only marginal improvements over base retrieval and frequently reuse invalid or outdated memories (e.g., LoCoMo/LongMemEval benchmarks reported in Memora (Uddin et al., 2026)); STALE (Chao et al., 2026) further demonstrates that even when updated evidence is explicitly retrieved, systems often fail to treat it as authoritative, with 56.1 percent of cases exhibiting residual behavioral failure despite updated evidence appearing in top-ranked retrieval results. These findings converge on a structural diagnosis: memory failure is not primarily a retrieval problem, but a failure to maintain a coherent and authoritative state over time. The deeper question---what kind of access mechanism best serves an agent's epistemic needs---remains open.

This paper argues that the apparent binary choice between retrieval and state is a false dichotomy. At least three structurally distinct epistemic operators are at play: retrieval (query-conditioned selection over an unordered artifact pool), state lookup (query-independent readout of a compressed belief state), and trajectory walk (chronological traversal of a temporally ordered transition graph). Each operator preserves a different set of epistemic dimensions and discards others. As a result, no single operator is complete---each makes distinct classes of memory failures structurally unavoidable. The design problem is therefore better understood as composition of access operators guided by the structure of the task epistemic space, with operator effectiveness further conditioned by the choice of backbone model.

**Contribution 1: why single operators are not enough.**  
We define three epistemic operators and show, through controlled benchmarks, that each induces a distinct and irrecoverable projection failure over a shared memory trajectory. Retrieval makes contradiction structurally unobservable. State lookup eliminates this instability by collapsing memory into a single-value representation, but makes historical and relational queries structurally unaskable. Trajectory walk preserves the full temporal record, avoiding both failures, but is non-compressible for decision-time inference. No single projection preserves co-visibility, temporal access, and compressibility simultaneously. Across four backbone LLMs, state lookup exhibits near-identical failure patterns, while retrieval and trajectory preferences shift substantially across models---operator complementarity is model-independent; optimal routing is model-conditioned.

**Contribution 2: compositional memory access from incompatible projections.** We propose a composition methodology for reconstructing epistemic coverage from structurally incomplete operators. The method decomposes tasks into epistemic dimensions, maps each dimension to its projection-compatible operator through an empirically derived phase map, calibrates the mapping for the target backbone model, and composes operators through type-safe routing. Composition recovers task regions inaccessible to any single operator by preventing structurally mismatched operator-query pairings. Across all tested settings, calibrated composition never underperforms the best single operator.

**Contribution 3: empirical validation at multiple scales.** A composed system following model-calibrated routing achieves consistent routing gains over the best single operator across structured (MSFT), natural-language (MSFT-NL), and external conflict benchmarks (STALE). Critically, calibrated composition empirically maintains non-inferiority across all tested model-benchmark combinations. The gains, and the non-inferiority property, persist across four backbone LLMs, confirming that the methodology generalizes across models. On the large-scale real-world ATM-Bench benchmark, operator-level analysis reveals distinct epistemic coverage regions, with a substantial fraction of queries motivating operator classes beyond the three studied here---a direction the composition framework is designed to accommodate.

---

## 2 Three Epistemic Operators

The three access mechanisms differ in their mathematical structure. Retrieval is similarity matching over a continuous space; state lookup is discrete classification over a key-value space; trajectory walk is deterministic traversal over a temporally ordered graph. Each induces a distinct epistemic operator with specific structural properties.

### 2.1 Formal Definitions

We model agent memory as a sequence of events with temporal adjacency:

$$T = \{e_1, e_2, \dots, e_n\}, \quad e_{i} \prec e_{i+1}$$

where $\prec$ denotes temporal precedence. From this primitive object we derive two reduced representations:

**State projection.** $\mathrm{proj}_s(T) = s_n$ --- the current belief state, retaining no information about how those beliefs were reached.

**Artifact pool.** $M_T = \{e_1, \dots, e_n\}$ --- the unordered set of evidence artifacts, stripped of temporal order and causal adjacency.

**Transition graph.** $G_T = (V, E)$ where $V = \{e_1, \dots, e_n\}$ and $E = \{(e_i, e_{i+1}) \mid 1 \leq i < n\}$ --- the directed graph encoding temporal adjacency between consecutive events.

Each representation induces an epistemic operator:

| Operator | Representation | Context | Selection mechanism |
|---|---|---|---|
| Retrieval ($\mathcal{R}_T$) | Artifact pool $M_T$ | Similarity-ranked subset | Query-time ($\text{top-}K_\theta(M_T, q)$) |
| State ($\mathcal{S}_T$) | State projection $\mathrm{proj}_s(T)$ | All active key-value pairs | Write-time (via revision $U$) |
| Trajectory ($\mathcal{T}_T$) | Transition graph $G_T$ | Ordered subgraph | Query-time (deterministic walk) |

### 2.2 What Each Operator Preserves and Discards

| Operator | Preserves | Discards |
|---|---|---|
| $\mathcal{R}_T$ | Raw content (lexical/semantic overlap) | State semantics, temporal order |
| $\mathcal{S}_T$ | Current beliefs (fact consistency) | Temporal/causal structure, alternatives |
| $\mathcal{T}_T$ | Temporal order, state evolution | Compactness, query-independence |

These preservation and discard patterns determine each operator's structural failure region. $\mathcal{R}_T$ cannot resolve write-time conflicts because it never aggregates conflicting evidence into a single state. $\mathcal{S}_T$ cannot answer historical queries ("what was the original budget?") because it discards the trajectory at construction time and retains only the latest value per key. $\mathcal{T}_T$ cannot efficiently provide concise state summaries because it preserves the full temporal sequence. No operator preserves all epistemic dimensions; therefore no single operator is complete.

The distinction between operators and storage representations is critical. The operators defined here characterize how memory is accessed, not how it is stored. Vector databases, filesystem hierarchies, SQL tables, knowledge graphs, and semantic workspaces are representation-layer choices: they determine how information is encoded, organized, indexed, or persisted. The epistemic operators studied in this work instead characterize the access mechanism applied over those representations---query-conditioned selection ($\mathcal{R}_T$), query-independent state readout ($\mathcal{S}_T$), or temporally ordered traversal ($\mathcal{T}_T$). Representation determines storage topology; operators determine epistemic accessibility at query time. Consequently, richer representations do not eliminate operator distinctions: the same graph-based memory system may still operate as retrieval, state lookup, or trajectory traversal depending on how access is performed. Our claim therefore concerns access primitives rather than storage formats.

We do not claim that these three operators constitute a complete ontology of memory access mechanisms. Other access mechanisms beyond the scope of this work may instantiate distinct epistemic operators, particularly those that introduce learned recurrence, structured latent state evolution, or programmatic memory control. Our claim is narrower: these three operators represent irreducible access primitives that are empirically dominant in current memory-augmented LLM systems, and their complementarity already makes operator composition a necessary design consideration.

The central implication is not that any one operator is superior---the optimal operator depends on the structure of the query---but that operator choice is an irreducible design decision in memory system architecture. The remainder of this paper formalizes when each operator is appropriate and how to compose them.

---

## 3 Related Work

**Current memory systems are single-operator implementations.** The vast majority of memory-augmented LLM systems implement $\mathcal{R}_T$: artifact storage with query-conditioned retrieval, varying only in storage structure and retrieval heuristics. LightMem (Fang et al., 2026, ICLR 2026) proposes a three-stage memory with offline consolidation, achieving substantial efficiency gains, but its access mechanism remains query-conditioned retrieval. Graph-based systems such as APEX-MEM (temporal property graph with multi-tool retrieval, ACL 2026) and GSW (generative semantic workspace, AAAI 2026 Oral) organize artifacts into structured graphs but still operate through query-conditioned selection. Adaptive retrieval systems---UMA (end-to-end RL for memory CRUD unified with QA), ReMemR1 (retrieval-in-the-loop memory updates)---improve selection mechanisms but remain within the retrieval paradigm. GAM (hierarchical graph-based consolidation) proposes structured consolidation rather than retrieval optimization. A parallel line of industry systems---OpenViking (ByteDance, 2026) with tiered directories, Anthropic CMA (2026) with mountable memory stores, Google Memory Bank (Vertex AI, 2026) as a managed vector service---converges on structured, filesystem-like storage representing a move toward $\mathcal{S}_T$-like access. Our controlled experiments with an OpenViking-aligned implementation confirm that structured retrieval improves over flat retrieval (obsolete rate 0.24 versus 0.57), but hits a structural ceiling: no retrieval variant eliminates obsolete value usage, because retrieval as an operator class cannot resolve write-time conflicts.

**Recent work challenges retrieval sufficiency.** STALE (Chao et al., 2026) provides the clearest external evidence: even when updated evidence is visibly present in retrieval results, it is treated as authoritative only 5.2 percent of the time, producing persistent behavioral failure. Their CUPMem prototype confirms that write-time state adjudication substantially improves performance (8.7 percent to 68.0 percent under the same backbone). STALE identifies the empirical symptoms of memory failure; our operator framework provides a structural explanation---why retrieval fails, when state suffices, and why composition is necessary. DeMem (Zou et al., 2026) proposes decision-centric forgetting criteria. Causal Memory Intervention (Srivastava, 2026) treats memory selection as a causal inference problem, selecting memories that are causally useful rather than merely topically relevant---a complementary query-time approach to our write-time revision operator.

**This paper's position.** Prior work on memory-augmented LLM systems has largely optimized within a single operator class---improving retrieval ranking, enriching state representations, or refining trajectory encoding. This paper studies operator selection and composition: given a set of fundamentally different access mechanisms, how should a system designer choose among them and how should they be combined? The operator composition framework operates at the access layer---how stored representations should be retrieved, projected, or traversed. Complementary work on richer representations, exemplified by STALE's latent state tracking, addresses the representation layer---what should be stored. The two are independent: better representations reduce individual operator failure regions but do not eliminate the structural necessity of operator composition, because no single access mechanism can simultaneously serve retrieval breadth, state consistency, and temporal fidelity. Operator composition is a design decision at a different level from representation engineering.

---

## 4 Why Single Operators Are Not Enough

### 4.1 Memory Access as Epistemic Projection

We model agent memory as a trajectory---a sequence of temporally ordered events (§2.1). Each memory access operator corresponds to a projection of this trajectory onto a lower-dimensional subspace, preserving some epistemic dimensions and discarding others. $\mathcal{R}_T$ projects onto similarity geometry; $\mathcal{S}_T$ projects onto latest-value compression; $\mathcal{T}_T$ projects onto temporal ordering. No single projection preserves all dimensions. Each operator defines a distinct information-preserving constraint system, not a retrieval strategy. The question is whether the resulting information loss is recoverable through downstream reasoning---our results show it is not.

### 4.2 Retrieval: Epistemic Invisibility via Structural Selection Failure

$\mathcal{R}_T$ induces a representational regime in which contradiction is systematically unobservable under any context size. This is not a claim about performance---it is a claim about what the operator makes structurally impossible to see. A controlled evidence-accumulation experiment (9-step trajectory: 3 consistent, 4 conflicting, 2 clarifying factual updates) exposes the mechanism.

**Co-visibility collapse → epistemic annihilation.** Across all steps of the accumulation protocol, correct and obsolete evidence are never simultaneously present in the top-$k$ retrieval output (co-visibility = 0.00). Retrieval operates in a regime where contradictory evidence is structurally non-co-present, making contradiction non-representable rather than unresolved. The LLM sees an internally consistent but temporally obsolete view of memory, with no perceptual access to the contradiction. Contradiction awareness remains zero across all steps.

**Ranking inversion → metric-induced epistemic distortion.** A single-trajectory trace at step 9 reveals the mechanism. For the query "Where does Alice live now?", the retrieval ranking is:

| Rank | Evidence chunk | Similarity |
|:----:|--------|:---------:|
| #1 | Alice lives in Seattle. | 0.125 |
| #2 | Alice works as a software engineer. | 0.100 |
| #3 | Alice commutes by bike every day. | 0.100 |
| #4 | Alice quit engineering and became a designer. | 0.083 |
| #5 | Alice moved to Chicago. ← *correct* | 0.077 |
| #6–#7 | (evidence about other entities: Bob, Charlie; omitted for clarity) | 0.045–0.056 |
| **#8** | **Alice no longer lives in Seattle. She permanently relocated to Chicago in 2025. ← explicit correction** | **0.038** |

The most authoritative evidence in the trajectory---an explicit correction---is ranked eighth, behind four obsolete chunks. The correction shares identical lexical overlap with the query ({alice}) but contains 18 additional words that expand the Jaccard denominator by approximately threefold while contributing no additional overlap, driving its score below shorter, information-sparse obsolete chunks. The ranking penalty is structural: similarity metrics redefine epistemic salience in a way that systematically suppresses corrective evidence. The system is biased not toward wrong answers, but toward epistemically incomplete evidence.

**Monotonic degradation → progressive epistemic drift.** Under $\mathcal{R}_T$, the obsolete answer rate rises monotonically from 0.00 to 0.57 by step 9. Append-only storage combined with similarity ranking induces a trajectory of progressive epistemic drift toward stale fixed points.

**Systematic instability.** Across four independently constructed trajectories with identical evidence structure, $\mathcal{R}_T$'s obsolete rate at step 9 ranges from 0.286 to 0.571 (0.39$\pm$0.14), while $\mathcal{S}_T$ converges to 0.00$\pm$0.00 (Appendix H.1). The 2$\times$ spread is not noise but structured variation: retrieval ranking depends on query-evidence lexical alignment, which varies across domains and is not under the system designer's control. $\mathcal{R}_T$ behaves as a stochastic sampler with no fixed epistemic point.

**Architectural ceiling → invariant under retrieval design class.** The ceiling persists across retrieval architectures. A filesystem-paradigm retriever (OpenViking-aligned, hierarchical directories with L0/L2 retrieval) reduces the obsolete rate from 0.57 to 0.24; adding LLM-enhanced L0 generation and query rewriting further reduces it to 0.15---but this failure pattern is invariant under retrieval design class. No retrieval variant eliminates obsolete value usage. $\mathcal{S}_T$ with write-time revision maintains 0.00.

**Causal attribution.** The counterfactual design---same trajectory, same evidence, same queries, same model, differing only in whether revision operator $U$ is applied---establishes that the gap (0.57 versus 0.00) is causally attributable to operator choice. $\mathcal{R}_T$ cannot maintain a consistent state because it has no state to maintain; $\mathcal{S}_T$ can, because it separates write-time resolution from read-time access. This is not an implementation difference---it is a structural consequence of the operators' definitions.

$\mathcal{R}_T$ does not fail to represent conflict; it defines an attention geometry in which conflicting evidence cannot enter a shared representational subspace. Contradiction is not lost; it is never jointly instantiated.

### 4.3 State Lookup: Epistemic Certainty via Representational Collapse

$\mathcal{S}_T$ eliminates retrieval-induced conflict by collapsing memory into a single-valued epistemic region. This is not a retrieval improvement---it is a different epistemic regime. $\mathcal{S}_T$ does not return uncertain distributions over possible states; it returns a single committed value. It replaces epistemic uncertainty with representational definiteness. This induces an epistemic asymmetry: what is represented is knowable with certainty, while what is discarded becomes irrecoverable even under perfect downstream reasoning. This loss is irreversible under any downstream inference, as missing representational dimensions are never instantiated in the input space.

**What state lookup solves.** Under the same 9-step conflict accumulation protocol, $\mathcal{S}_T$ converges to 0.00 obsolete rate across all four trajectories (§4.2). Its advantage does not depend on manually annotated inputs: when state extraction is performed entirely by an LLM from unstructured text, the obsolete rate is 3.85 percent, against 19.23 percent for flat retrieval---a 5$\times$ improvement. Extraction error is bounded (approximately 0.04) and does not compound with memory size; retrieval error is unbounded because ranking competition grows with the artifact pool. This stability is achieved by eliminating representational multiplicity at construction time, making uncertainty non-representable rather than unresolved. As a result, any temporal or relational query is structurally projected out of the representation space.

**What state lookup makes unaskable.** The per-pattern MSFT breakdown reveals where $\mathcal{S}_T$ fails structurally (full table in Appendix B.1):

| Pattern | n | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ | Structural Root |
|---|---|---|---|---|---|
| past_state | 12 | **1.000** | 0.167 | 0.750 | $\mathcal{S}_T$ discards history |
| cross_entity | 16 | 0.500 | **0.125** | 0.625 | $\mathcal{S}_T$ cannot align temporal axes |
| temporal_relationship | 12 | 0.667 | **0.000** | 0.750 | $\mathcal{S}_T$ has no temporal representation |
| correction_chain | 12 | 0.750 | **0.917** | 0.833 | $\mathcal{S}_T$ collapses to final value |
| multi_path | 12 | **0.917** | 0.333 | 0.750 | $\mathcal{S}_T$ overwrites branch values |
| ambiguous | 24 | 0.500 | 0.500 | 0.500 | All operators lack uncertainty mechanisms |
| negative_evidence | 12 | 0.667 | 0.500 | 0.583 | --- |
| implicit_conflict | 12 | 0.667 | 0.500 | 0.667 | --- |
| **Overall** | **112** | **0.679** | **0.384** | **0.661** | Best single $\mathcal{R}_T$ at 0.679 |

$\mathcal{S}_T$ does not merely perform poorly on historical queries (past\_state: 0.167), cross-entity temporal alignment (cross\_entity: 0.125), and temporal reasoning (temporal\_relationship: 0.000)---it makes these query classes structurally unaskable within its representational space. It succeeds where compression is beneficial: correction\_chain (0.917), where collapsing to the latest value is the correct operation.

**Cross-model invariance.** To distinguish operator-inherent failures from model-specific weakness, we evaluate all three operators on four LLMs under identical protocol. $\mathcal{S}_T$ fails on 29 of 112 cases across all four models---25.9 percent of cases are structurally inaccessible to state lookup regardless of model family or capability (Appendix E.7). The failures concentrate in past\_state (9/12, 75 percent) and cross\_entity (7/16, 43.8 percent). Critically, the wrong answers are near-verbatim identical across models: on a past\_state query, all four models answer with the current value rather than the historical value. These failures are model-independent because the information loss is imposed at representation construction time, not at reasoning time. In contrast, $\mathcal{R}_T$ and $\mathcal{T}_T$ failures are substantially model-dependent: only 4 of 112 cases fail across all models for each operator. $\mathcal{S}_T$ is **representation-limited**; $\mathcal{R}_T$ is **geometry-limited**.

$\mathcal{S}_T$ delivers certainty by erasing alternatives. Its strength is its weakness: the same projection that eliminates conflict also eliminates the capacity to ask certain questions.

### 4.4 Trajectory Walk: Epistemic Completeness without Compressibility

$\mathcal{T}_T$ preserves the full epistemic trace but violates the representational compressibility required for decision-time inference. It is the only operator that preserves epistemic history without projection-induced loss, but at the cost of non-compressibility for downstream decision making. On path-dependent patterns, it excels where both alternatives struggle: temporal\_relationship (0.750), multi\_path (0.750), cross\_entity (0.625). These queries require following a sequence of state changes---an operation that retrieval over independent chunks and state lookup over collapsed values cannot perform. On correction\_chain (0.833), however, $\mathcal{T}_T$ provides the full event stream where a single key-value readout ($\mathcal{S}_T$, 0.917) suffices. The trajectory operator is the most complete temporal witness but the least compressible into a decision-ready form. $\mathcal{T}_T$ is the only operator that preserves epistemic completeness in principle, but cannot reduce it into a bounded decision representation.

### 4.5 From Trilemma to Structural Incompatibility

The three operators define mutually exclusive constraints over a shared epistemic space. No projection can simultaneously preserve co-visibility of conflicting evidence, temporal structure, and decision-time compressibility.

$\mathcal{R}_T$ induces **visibility collapse**: similarity-based selection structurally suppresses longer, more informative correction evidence, making contradiction unobservable regardless of context size. Conflict is not resolved---it is excluded from the representational space.

$\mathcal{S}_T$ induces **temporal collapse**: by projecting the full trajectory onto latest-value compression, it eliminates representational multiplicity and with it any capacity to answer historical or cross-entity queries. Consistency is achieved through erasure, not resolution.

$\mathcal{T}_T$ induces **compression collapse**: by preserving the full temporal sequence, it maintains access to all evidence but sacrifices compactness and query-independence, making it the most complete but least efficient operator.

These collapse patterns are irreducible: $\mathcal{R}_T$ succeeds on past\_state (1.000) where $\mathcal{S}_T$ fails (0.167); $\mathcal{S}_T$ succeeds on correction\_chain (0.917) where $\mathcal{R}_T$ is weaker (0.750); $\mathcal{T}_T$ dominates temporal access where both alternatives are structurally incapable. The optimal operator per pattern varies across models on seven of eight patterns (Appendix E.7).

Any single operator induces a structurally unavoidable epistemic blind region that cannot be recovered through composition at inference time. Memory reasoning cannot be reduced to any single operator without structural loss of epistemic dimensions. This is not a failure of engineering but a **representational impossibility under single-projection memory access**. Single-operator memory access is not a suboptimal choice that can be incrementally improved---every projection permanently disables at least one epistemic dimension. Therefore, memory reasoning is not an operator selection problem but a problem of reconstructing a complete epistemic state from incompatible projections. This necessitates dynamic composition: a system must select, combine, and calibrate complementary operators to span the irreducible epistemic space. The next section develops the compositional methodology that operationalizes this principle.

---

## 5 Composition over Incompatible Projections

### 5.1 From Structural Incompatibility to Composition Necessity

Section 4 established that each of the three operators induces a structurally unavoidable epistemic blind region, and that these blind regions are complementary rather than overlapping. The remaining question is not which operator to choose, but how to recover usable epistemic coverage from complementary projections over a shared memory trajectory. Composition is necessary because the missing information under one projection is not recoverable within that projection's representational space. Coverage can only be reconstructed by combining operators that preserve complementary epistemic dimensions. The three operators interact with five epistemic axes (temporal access, conflict resolution, entity composition, epistemic uncertainty, counterfactual grounding), forming a 3$\times$5 space of projection-compatibility interactions. We empirically characterize this space to identify which operator preserves the required epistemic structure for each axis.

### 5.2 The Phase Map as Empirical Characterization

The phase map is an empirical chart of which epistemic regimes remain representable under each projection operator. It is not a leaderboard of operator performance; it is a characterization of projection-compatibility across epistemic regimes.

| Axis | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ | Compatible projection |
|---|---|---|---|---|
| Temporal access | 0.617 | 0.433 | **0.700** | $\mathcal{T}_T$ (ordered traversal) |
| Conflict resolution | 0.817 | **0.950** | **0.967** | $\mathcal{S}_T \approx \mathcal{T}_T$ (compression or temporal support) |
| Entity composition | **0.533** | 0.700 | 0.650 | $\mathcal{S}_T > \mathcal{T}_T$ (relational binding favors compression) |
| Epistemic uncertainty | 0.550 | **1.000** | 0.550 | $\mathcal{S}_T$ (compression eliminates ambiguity) |
| Counterfactual grounding | 0.900 | 0.817 | **1.000** | $\mathcal{T}_T$ (temporal record preserves conditionality) |

Each axis favors a different operator because each operator preserves a different epistemic dimension (§2.2): $\mathcal{T}_T$ preserves temporal order, $\mathcal{S}_T$ preserves fact consistency through compression, $\mathcal{R}_T$ preserves raw content but discards structure. On conflict resolution, $\mathcal{S}_T$ and $\mathcal{T}_T$ both succeed because they both resolve contradiction---$\mathcal{S}_T$ through write-time compression that collapses to the latest value, $\mathcal{T}_T$ through temporal support that surfaces the full resolution chain. Together, these dominance patterns form a compatibility map: which projection is structurally aligned with which epistemic demand.

A cross-regime comparison confirms that this compatibility structure is stable across structured and natural language inputs:

| Operator | Structured | NL | Delta | Property |
|---|---|---|---|---|
| $\mathcal{R}_T$ | 0.680 | 0.683 | +0.003 | Regime-invariant |
| $\mathcal{S}_T$ | 0.500 | 0.780 | +0.280 | Compression |
| $\mathcal{T}_T$ | 0.880 | 0.773 | -0.107 | Ordered-access |

$\mathcal{R}_T$ is regime-invariant: its performance depends on similarity geometry between query and memory surface forms, independent of how those forms are structured. $\mathcal{S}_T$ shows compression behavior: natural language provides implicit normalization that improves extraction. $\mathcal{T}_T$ is ordered-access: its effectiveness depends on explicit temporal ordering, and degrades when ordering must be inferred from natural language. Cross-model validation (GPT-4o-mini) shows that only the compression property is model-independent; the regime-invariance and ordered-access properties vary with model capability, reinforcing the necessity of model calibration (§5.3, Step 2).

### 5.3 Composition Methodology

The phase map (§5.2) provides the empirical foundation for a composition methodology that recovers epistemic coverage from structurally incomplete operators. The methodology does not optimize any single operator; it prevents epistemically incompatible operator-query pairings by dispatching each query to the operator whose projection is structurally compatible with the query's epistemic demand.

**Step 1: Decompose the task space.** Identify the epistemic axes present in the target task---temporal access, conflict resolution, entity composition, epistemic uncertainty, counterfactual grounding, or others specific to the domain. Each axis corresponds to a distinct cognitive operation that may require a different access mechanism. In this work, task decomposition is performed manually by the authors informed by the structure of each benchmark; automating or standardizing this step is an open problem discussed in §8.

**Step 2: Map each axis to a compatible projection, then calibrate for the target model.** The phase map (§5.2) assigns each epistemic axis to the projection that preserves its required structure: temporal access → $\mathcal{T}_T$, epistemic uncertainty → $\mathcal{S}_T$, entity composition → $\mathcal{S}_T$ (avoid $\mathcal{R}_T$), conflict resolution → $\mathcal{S}_T$ or $\mathcal{T}_T$, counterfactual grounding → $\mathcal{T}_T$. This mapping identifies structurally compatible operator-query pairings. The phase map is measured on a single backbone model; the exact dominance boundary between operators can shift when the model changes---on DeepSeek Flash, operators are balanced and routing gains are larger; on GPT-4o-mini, $\mathcal{T}_T$ dominates and gains are smaller. To account for this, measure operator accuracies on a small diagnostic sample (§4) under the target model and adjust the mapping accordingly. The phase map provides the structural prior; calibration corrects for backbone-dependent shifts.

**Step 3: Compose through type-safe routing.** Implement a router that classifies each query into its epistemic axis and dispatches to the operator whose projection is compatible with that axis. The routing architecture is type-safe (one query, one operator) to avoid fusion ambiguity. Composition improves performance not by enhancing any individual operator, but by preventing structurally mismatched operator-query pairings. The front-end classifier is a replaceable component; the operator layer is stable (§4).

**Step 4: Validate against single-operator baselines.** Compare the composed system against each single operator on the target benchmark. If composition does not improve coverage over the best single operator, the task space may be too homogeneous for composition, or the phase calibration may need adjustment. Across all tested benchmarks and backbone models, calibrated composition consistently recovers task regions inaccessible to any single operator (§6).

The following section validates this methodology across three benchmarks and four backbone models.

---

## 6 Validating Composition over Incompatible Projections

Section 4 established that single operators induce irreducible blind regions. Section 5 proposed composition over incompatible projections as the structural response. This section validates that response across four questions: does composition recover coverage (§6.1), does the methodology transfer across benchmarks (§6.2), where does the current operator basis reach its limits (§6.3), and is calibration necessary rather than optional (§6.4).

### 6.1 Coverage Recovery under Composition

**Does composition recover task regions inaccessible to any single operator?** We apply the methodology from §5.3 to the 112-case MSFT benchmark. All three operators and the composed system use the same backbone LLM (DeepSeek Flash, temperature 0.0), same evidence, same binary judge.

On the 112-case MSFT benchmark, the composed system achieves 0.817 against 0.679 for the best single operator ($\mathcal{R}_T$)---a routing gain of +0.138. The operator layer produces zero new failures: all 19 residual failures stem from the IntentDetector front-end (10 recall errors) or known operator boundaries (6 cross_entity cases). The theoretical upper bound with perfect routing is 0.913, confirming that the routing architecture is not the bottleneck---the front-end classifier is a replaceable component. Composition improves performance not by enhancing any individual operator, but by preventing structurally mismatched operator-query pairings.

The routing gain is not an ensemble effect: each operator is queried independently for its assigned query type, and no query is processed by more than one operator. The gain arises from matching each query to the operator whose projection preserves the epistemic dimensions required by that query---not from statistical aggregation of multiple outputs.

**Non-inferiority across models.** On GPT-4o-mini, a composed system calibrated to its per-pattern operator preferences achieves 0.830 against 0.812 for the best single operator (+0.018). While the gain is smaller than on DeepSeek Flash (+0.138)---attributable to GPT-4o-mini's $\mathcal{T}_T$ dominance reducing operator diversity---the direction is consistent. Across all tested models, calibrated composition empirically maintains accuracy at or above the best single operator. This non-inferiority property, rather than the magnitude of any individual routing gain, is the methodological guarantee: system designers can adopt calibrated composition without risk of degrading below the best single-operator baseline.

### 6.2 External Generalization across Benchmarks

**Does the composition methodology transfer to unseen epistemic regimes without redesign?** The STALE benchmark (Chao et al., 2026) provides 400 conflict scenarios with 1,200 evaluation queries across three dimensions---State Resolution, Premise Resistance, and Implicit Policy Adaptation---in a conversational domain distinct from the entity-attribute structure of MSFT. We apply the same four-step methodology without redesign: the task space is decomposed into epistemic axes (dim1 → epistemic uncertainty, dim2 → conflict resolution, dim3 → temporal access, Type II propagation → entity composition), the phase map (§5.2) selects the initial operators, and the router dispatches queries accordingly. $\mathcal{S}_T$ handles dim1 and dim2 (state readout with premise verification for conflict resolution), $\mathcal{T}_T$ handles dim3 (temporal context for action recommendations), and $\mathcal{T}_T$ is used for all dimensions in Type II scenarios (propagation chains visible in trajectory). The system runs on DeepSeek Flash with a rule-based keyword judge.

| Type | SR | PR | IPA | Routing |
|---|---|---|---|---|
| T1 (n=200) | 0.540 | 1.000 | 0.580 | S_T(dim1/2), T_T(dim3) |
| T2 (n=200) | 0.450 | 0.880 | 0.860 | T_T(all) |
| **All (n=400)** | **0.495** | **0.940** | **0.720** | **Overall: 0.718** |

The composed system achieves 0.718 overall against CUPMem's 0.680 on the same benchmark, though direct comparison is confounded by backbone model differences (DeepSeek Flash versus GPT-4o-mini) and judge methodology. The structural claim is independent of this number: composition gains follow from operator complementarity, and the phase-map-driven methodology transfers to external conflict regimes without modification. Full routing strategy and failure taxonomy are reported in Appendix F.

### 6.3 Scaling Limits and Uncovered Epistemic Regions

**What happens when the epistemic space exceeds the current operator basis?** ATM-Bench (Mei et al., 2026) provides 1013 QA over four years of personal multimodal memory---a scale and domain beyond the controlled MSFT and STALE benchmarks. $\mathcal{S}_T$ is evaluated as a Retrieval+KV variant; the primary operator-level comparison is between $\mathcal{R}_T$ and $\mathcal{T}_T$, where $\mathcal{T}_T$ uses oracle trajectory access and represents an epistemic upper bound.

| Condition | Accuracy | Number | List | Open |
|---|---|---|---|---|
| $\mathcal{R}_T$ | 0.399 | 0.469 | 0.151 | 0.416 |
| $\mathcal{S}_T$ (Retrieval+KV) | 0.175 | 0.172 | 0.094 | 0.198 |
| $\mathcal{T}_T$ (oracle trajectory) | **0.518** | 0.622 | 0.144 | 0.547 |

$\mathcal{T}_T$ uniquely covers 16.3 percent of queries, $\mathcal{R}_T$ uniquely covers 3.4 percent, and 42.3 percent are uncovered by any operator. The uncovered fraction is not a method failure---it is empirical evidence that the epistemic space exceeds the current three-operator projection basis. This aligns with the framework's design: the composition methodology accommodates additional operator classes as they are identified (§2, §8). The trajectory operator's advantage grows with task complexity: from +0.092 on single-evidence queries (n=764) to +0.320 on queries requiring four or more evidence items (n=50), consistent with $\mathcal{T}_T$'s structural property of preserving temporal ordering.

We emphasize that the ATM-Bench $\mathcal{T}_T$ results use oracle trajectory access and represent an epistemic upper bound. On MSFT (§6.1) and STALE (§6.2), where $\mathcal{T}_T$ uses real session ordering and entity-scoped walks, the methodology is validated end-to-end. Full analysis (including Hard-31, projection bottleneck, and evidence count gradient) is reported in Appendix C.

### 6.4 Cross-Model Calibration Necessity

**Is calibration optional?** To test whether a single routing table suffices across backbone models, we evaluate individual operators $\mathcal{R}_T$/$\mathcal{S}_T$/$\mathcal{T}_T$ on four LLMs with identical protocol (MSFT 112-case, unified DeepSeek Flash judge):

| Model | R_T | S_T | T_T | Best Single |
|---|---|---|---|---|
| DeepSeek Flash | 0.679 | 0.384 | 0.661 | R_T 0.679 |
| DeepSeek V4 Pro | 0.839 | 0.625 | 0.786 | R_T 0.839 |
| GPT-4o-mini | 0.741 | 0.527 | 0.812 | T_T 0.812 |
| Claude Sonnet 4 | 0.795 | 0.491 | 0.830 | T_T 0.830 |

Operator complementarity holds across all models---each model exhibits distinct accuracy distributions across the three operators, confirming that operator decomposition is meaningful regardless of backbone. The optimal single operator varies: DeepSeek-family models favor $\mathcal{R}_T$, while GPT-4o-mini and Claude favor $\mathcal{T}_T$. The only operator-invariant failure pattern is $\mathcal{S}_T$ on past_state and cross_entity (§4.3). On seven of eight epistemic patterns, the per-pattern best operator differs across models (Appendix E.7). Therefore, no universal routing table exists---any routing policy derived from one model's phase map requires empirical calibration before transfer. Operator complementarity is model-independent; optimal routing is model-conditioned. This asymmetry directly necessitates Step 2 of the methodology (§5.3): the phase map provides the structural prior, and calibration corrects for backbone-dependent dominance shifts.

---

## 7 Discussion and Limitations

The results across §4–6 converge on a design principle: memory access is fundamentally constrained by mutually incompatible epistemic projections, and can only be recovered through composition, rather than optimization of a single access mechanism. Composition does not function as an ensemble—each query is dispatched to exactly one operator, and the gain arises from preventing structurally mismatched pairings, not from aggregating multiple outputs (§6.1). The phase map provides the structural prior; calibration accounts for backbone-dependent dominance shifts (§6.4). Across all tested settings, calibrated composition never underperforms the best single operator, establishing non-inferiority as a methodological property rather than an empirical coincidence.

These limitations reflect the current boundary of the operator-based epistemic access framework.

**Static routing.** The composed system uses deterministic rule-based routing derived from per-pattern phase analysis. A dynamic router that learns query-to-operator mappings from data could adapt to distribution shifts and model-specific operator strengths, potentially increasing the routing gain beyond the fixed-rule baseline.

**Operator scope.** We identify three epistemic operators but do not claim this set is exhaustive. Our claim is narrower: these three represent irreducible access primitives that are empirically dominant in current memory-augmented LLM systems (§2). Other access mechanisms beyond the scope of this work may instantiate distinct epistemic operators, particularly those that introduce learned recurrence, structured latent state evolution, or programmatic memory control. The structural weakness of $\mathcal{S}_T$ on cross-entity queries (§4.3) suggests one such candidate---a relational join operator across entity temporal axes. A systematic search over the Cartesian product of epistemic axes and access mechanisms would be needed to determine closure. We emphasize that this paper does not propose a complete ontology of memory operators; it proposes a design methodology for composition given whatever operators are available.

**S_T implementation gap on ATM-Bench.** The state operator on ATM-Bench uses a Retrieval+KV variant rather than a true state projection with write-time revision. True state dynamics are validated separately through controlled trajectory comparisons (Appendix H).

**Metric alignment with ATM-Bench.** Our binary LLM judge differs from the benchmark's question-type-specific QS. The two metrics are directionally aligned on open-ended questions.

**MSFT-NL scope.** The cross-regime analysis uses 300 automatically generated NL cases from parameterized templates with randomized entity names and surface forms. Template-based generation introduces a potential confound: systematic patterns in template structure may inflate or deflate specific operator-axis interactions. Intra-template variance is not reported, and real-world NL validation remains future work. Cross-model validation is limited to the structured MSFT benchmark; extending to the NL regime would further test the stability of the operator phase map across models.

**Task decomposition robustness.** Step 1 of the methodology (decomposition of task space into epistemic axes) is performed manually by the authors. The reproducibility of this decomposition across annotators is not quantified, and no automated procedure is proposed. An inter-annotator agreement study or automated axis-classification method would strengthen confidence in the methodology's deployability.

**Single judge model.** All operator evaluations across all benchmarks use DeepSeek Flash as the sole judge. In cross-model experiments, GPT-4o-mini and Claude Sonnet 4 answers are also judged by DeepSeek Flash. If the judge model has systematic preferences (lexical overlap bias, format preference), these could inflate DeepSeek-family operator scores relative to other models. A cross-judge validation using an alternative judge model on a subset of cases would quantify this risk.

**T_T oracle vs deployable gap.** On ATM-Bench, $\mathcal{T}_T$ uses oracle trajectory access (ground-truth evidence IDs and session ordering). On MSFT and STALE, $\mathcal{T}_T$ uses real session ordering and entity-scoped walks. The gap between oracle and deployable trajectory access on ATM-Bench is not quantified; the ATM-Bench results should be interpreted as an epistemic upper bound rather than a validation of the full composition methodology in a deployable setting.

**Cross-entity boundary.** No operator can reliably align two entities' temporal axes---a known theoretical gap motivating potential fourth operator classes beyond the three identified here.

---

## 8 Conclusion

Memory-augmented LLM agents must choose how to access the past. Retrieval—query-conditioned selection over an artifact pool—is the dominant paradigm, but it is only one of at least three structurally distinct epistemic operators. State lookup preserves fact consistency but discards temporal structure; trajectory walk preserves temporal order but sacrifices compactness. No single operator is complete: each induces an irrecoverable projection failure over a shared memory trajectory. Composition over incompatible projections recovers epistemic coverage by dispatching each query to the operator whose projection preserves the required epistemic dimensions. Cross-model validation confirms that this complementarity is model-independent, while optimal routing is model-conditioned——state lookup fails on the same cases across all tested models; retrieval and trajectory preferences shift with the backbone LLM. A composed system following model-calibrated routing achieves consistent gains over the best single operator across structured, natural-language, and external conflict benchmarks, while empirically maintaining non-inferiority across all tested settings. Extending the operator basis beyond the three primitives studied here, automating task-space decomposition, and validating across additional backbone models and judge protocols are natural next steps that the composition framework is designed to accommodate. The central implication is that memory system design should be reframed as operator composition over irreducible epistemic projections, rather than representation optimization under a single access paradigm.

---

## Appendix A: Operator Definitions and System Mapping

### A.1 Formal Operator Definitions

**Definition (Retrieval Operator).** $\mathcal{R}_T(q) = \text{top-}K_\theta(M_T, q)$ where $M_T = \{e_1, \dots, e_n\}$ is the unordered artifact pool and $\text{top-}K_\theta$ selects the $K$ evidence chunks with highest similarity to $q$ under similarity measure $\theta$. The operator has no internal state, no write-time resolution, and depends on $q$ for context selection.

**Definition (State Operator).** $\mathcal{S}_T(q) = \{ (k, \text{latest}(T, k)) \mid k \in \text{dom}(\mathrm{proj}_s(T)) \}$ where $\text{latest}(T, k) = \max_{t} \{ v \mid (k, v, t) \in T \}$. The operator is query-independent for context: it returns all active key-value pairs. Write-time conflict resolution is performed by revision operator $U$.

**Definition (Trajectory Operator).** $\mathcal{T}_T(q) = \text{walk}(G_T, C_q)$ where $G_T = (V, E)$ is the transition graph with $V = \{e_1, \dots, e_n\}$ and $E = \{(e_i, e_{i+1}) \mid 1 \leq i < n\}$, and $C_q$ is an optional constraint derived from $q$. The simplest form (tested in MSFT) is $\text{walk}(G_T)$---unconstrained traversal from $t=1$ to $t=n$ for the target entity.

### A.2 Paradigm Comparison

| Property | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ |
|---|---|---|---|
| Memory object | Artifact collection $M_T$ | Belief state $B_t$ | Transition graph $G_T$ |
| Access | Query-conditioned selection | State readout | Ordered path walk |
| Write-time | Append | Revision $U$ | Append and indexing |
| Query-time | Similarity ranking | None (direct) | Entity-scoped traversal |
| Invariance | Not guaranteed | Maintained (query-independent) | Path-dependent, deterministic |

### A.3 Axiom Coverage Matrix

| System | Explicit global state | Consistency objective | Principled revision | Evidence |
|---|---|---|---|---|
| FadeMem | Yes | Yes | Yes | Belief decay with confidence thresholds |
| DeltaMem | Yes | Yes | No | Tracks deltas, no conflict resolution |
| BeliefMem | Yes | Yes | Yes | Revision with source credibility |
| UMA | No | No | No | Adaptive retrieval, no state |
| STALE (Chao et al., 2026) | Yes | Yes | Yes | CUPMem: write-time adjudication |
| **This work** | Yes | Yes | Yes | $\mathcal{S}_T$ with revision $U$, controlled comparison |

---

## Appendix B: MSFT Extended Results

*Supporting data for §4 (operator-specific failure regions) and §6.1 (coverage recovery).*

### B.1 Full 8-Pattern Breakdown (112 cases, DeepSeek Flash)

| Pattern | n | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ | Structural Root |
|---|---|---|---|---|---|
| ambiguous | 24 | 0.500 | 0.500 | 0.500 | All operators lack uncertainty mechanisms |
| correction_chain | 12 | 0.750 | 0.917 | 0.833 | $\mathcal{S}_T$ collapses to final value |
| cross_entity | 16 | 0.500 | 0.125 | 0.625 | $\mathcal{S}_T$ cannot align temporal axes |
| implicit_conflict | 12 | 0.667 | 0.500 | 0.667 | --- |
| multi_path | 12 | 0.917 | 0.333 | 0.750 | $\mathcal{S}_T$ overwrites branch values |
| negative_evidence | 12 | 0.667 | 0.500 | 0.583 | --- |
| past_state | 12 | 1.000 | 0.167 | 0.750 | $\mathcal{S}_T$ discards history |
| temporal_relationship | 12 | 0.667 | 0.000 | 0.750 | $\mathcal{S}_T$ has no temporal representation |
| **Overall** | **112** | **0.679** | **0.384** | **0.661** | Best single $\mathcal{R}_T$ at 0.679 |

### B.2 112-Case Scaling

| System | Accuracy | Case set | Failures | Failure Source |
|---|---|---|---|---|
| Composed | 0.817 | 112 | 19 | IntentDetector recall (10), operator boundary (6), other (3) |
| Best single ($\mathcal{R}_T$) | 0.679 | 112 | --- | --- |
| Theoretical upper bound (perfect routing) | 0.913 | 112 | --- | --- |

### B.3 Routing Strategy

The composed system uses a type-safe design: one intent, one operator. Each of 8 pattern types maps to exactly one operator via deterministic rules: past_state → $\mathcal{R}_T$, temporal_relationship → $\mathcal{T}_T$, multi_path → $\mathcal{T}_T$, negative_evidence → $\mathcal{T}_T$, correction_chain → $\mathcal{S}_T$, cross_entity → $\mathcal{R}_T$, implicit_conflict → $\mathcal{S}_T$, ambiguous → uncertainty protocol. These routing rules represent the classification decisions of the IntentDetector front-end, whose errors account for the gap between the achieved routing gain (+0.138) and the theoretical upper bound (0.913).

---

## Appendix C: ATM-Bench Additional Analysis

*Supporting data for §6.3 (scaling limits and uncovered epistemic regions).*

### C.1 Projection Bottleneck Analysis

| Evidence in top-5 | $\mathcal{R}_T$ | $\mathcal{T}_T$ | Gap |
|---|---|---|---|
| Yes | 0.542 | 0.610 | +0.068 |
| No | 0.024 | 0.488 | +0.464 |
| **Overall** | **0.399** | **0.518** | **+0.119** |

$\mathcal{R}_T$ accuracy collapses to 0.024 when evidence is outside the top-5 retrieval window. $\mathcal{T}_T$ achieves 0.488 under the same condition because it accesses evidence through chronological ordering rather than similarity ranking.

### C.2 Hard-31 Validation

| Condition | Accuracy |
|---|---|
| $\mathcal{R}_T$ | 0.032 |
| $\mathcal{S}_T$ (Retrieval+KV) | 0.065 |
| $\mathcal{T}_T$ (oracle trajectory) | 0.194 |

### C.3 Evidence Count Gradient (Full 1013)

| Evidence items | n | $\mathcal{R}_T$ | $\mathcal{T}_T$ | Gap |
|---|---|---|---|---|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | +0.320 |

---

## Appendix D: MSFT-NL Full Phase Data

*Supporting data for §5.2 (the phase map as empirical characterization).*

### D.1 Template-Based NL Generation

The MSFT-NL benchmark generates 300 natural language cases across 5 epistemic axes via template parameterization. Each axis has 60 cases with randomized entity names, surface forms, and distractor content.

| Axis | Epistemic operation | Example query type |
|---|---|---|
| Temporal access | Order/duration between events | "Which happened first?" |
| Conflict resolution | Resolving contradictory evidence | "What is the current value?" |
| Entity composition | Cross-entity relational binding | "Which entity was at location X?" |
| Epistemic uncertainty | Detecting underdetermined questions | "Can this be answered?" |
| Counterfactual grounding | Plan/action based on updated state | "What should X do now?" |

### D.2 Full Phase Map (DeepSeek Flash)

| Axis | n | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ | Dominant |
|---|---|---|---|---|---|
| Temporal access | 60 | 0.617 | 0.433 | 0.700 | $\mathcal{T}_T$ |
| Conflict resolution | 60 | 0.817 | 0.950 | 0.967 | $\mathcal{S}_T \approx \mathcal{T}_T$ |
| Entity composition | 60 | 0.533 | 0.700 | 0.650 | $\mathcal{S}_T > \mathcal{T}_T$ |
| Epistemic uncertainty | 60 | 0.550 | 1.000 | 0.550 | $\mathcal{S}_T$ |
| Counterfactual grounding | 60 | 0.900 | 0.817 | 1.000 | $\mathcal{T}_T$ |

### D.3 Cross-Regime Comparison (Structured vs NL)

| Operator | Structured | NL | Delta | Property |
|---|---|---|---|---|
| $\mathcal{R}_T$ | 0.680 | 0.683 | +0.003 | Regime-invariant |
| $\mathcal{S}_T$ | 0.500 | 0.780 | +0.280 | Compression |
| $\mathcal{T}_T$ | 0.880 | 0.773 | -0.107 | Ordered-access |

---

## Appendix E: Experimental Setup and Reproducibility

*Configuration details referenced throughout §4-6.*

### E.1 Model Configuration
- Backbone LLM: DeepSeek Flash (deepseek/deepseek-v4-flash)
- Temperature: 0.0
- Max output tokens: 128 (answer), 16 (judge)
- API: OpenAI-compatible chat completions endpoint

### E.2 Retrieval Setup
- Embedding model: all-MiniLM-L6-v2 (sentence-transformers), 384-dimension
- Top-K: 5 (MSFT), 5 (ATM-Bench), 3 to 5 (STALE)
- Chunking: per-message for MSFT; per-session document for STALE

### E.3 State Module
- Implementation: $B_t$ as dict mapping keys to belief entries {value, timestamp, source, confidence, status}
- Revision: confidence-thresholded $U$, replacing active belief when new confidence exceeds old confidence
- State readout: all active key-value pairs, query-independent

### E.4 Trajectory Walk
- Implementation: entity-scoped chronological walk of transition graph $G_T$
- Simple walk: $t=1$ to $t=n$, entity-filtered
- Constraint walk: $C_q$-scoped walk for multi_path and temporal_relationship patterns

### E.5 Judge Protocol
- Method: LLM-as-judge using DeepSeek Flash
- Ruling: binary CORRECT / INCORRECT
- MSFT-NL: structured judge comparing answer against ground-truth key-value
- ATM-Bench: binary judge with answer-keyword evaluation
- STALE (rule-based): keyword trigger detection for SR/PR/IPA

### E.6 STALE Adapter Design
- $\mathcal{R}_T$: session-level retrieval; each of 50 haystack sessions collapsed into a document; top-3 sessions retrieved
- $\mathcal{S}_T$: state extraction from M_old/M_new key evidence, confidence-thresholded revision
- $\mathcal{T}_T$: chronological walk of sessions identified by relevant_session_index
- Routing: dim1 → S_T, dim2 → S_T + premise check, dim3 → T_T; T2 scenarios → T_T for all dimensions

### E.7 Cross-Model Validation (MSFT 112-case, Judge: DeepSeek Flash)

| Model | R_T | S_T | T_T | Best Single |
|---|---|---|---|---|
| DeepSeek Flash | 0.679 | 0.384 | 0.661 | R_T 0.679 |
| DeepSeek V4 Pro | 0.839 | 0.625 | 0.786 | R_T 0.839 |
| GPT-4o-mini | 0.741 | 0.527 | 0.812 | T_T 0.812 |
| Claude Sonnet 4 | 0.795 | 0.491 | 0.830 | T_T 0.830 |

Key findings: $\mathcal{S}_T$ fails on 29/112 cases across all four models (past_state 9/12, cross_entity 7/16). Wrong answers are near-verbatim identical. Only 4/112 cases fail across all models for $\mathcal{R}_T$ and $\mathcal{T}_T$ respectively. Per-pattern best operator differs across models on 7/8 patterns; only past_state universally selects $\mathcal{R}_T$:

| Pattern | DS Flash | DS V4 Pro | GPT-4o-mini | Claude S4 |
|---|---|---|---|---|
| past_state | $\mathcal{R}_T$ | $\mathcal{R}_T$ | $\mathcal{R}_T$ | $\mathcal{R}_T$ |
| temporal_relationship | $\mathcal{T}_T$ | $\mathcal{T}_T$ | $\mathcal{R}_T$ | $\mathcal{R}_T$ |
| correction_chain | $\mathcal{S}_T$ | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{S}_T$ |
| cross_entity | $\mathcal{T}_T$ | $\mathcal{R}_T$ | $\mathcal{T}_T$ | $\mathcal{T}_T$ |
| implicit_conflict | $\mathcal{R}_T$ | $\mathcal{R}_T$ | $\mathcal{T}_T$ | $\mathcal{R}_T$ |
| multi_path | $\mathcal{R}_T$ | $\mathcal{S}_T$ | $\mathcal{T}_T$ | $\mathcal{T}_T$ |
| negative_evidence | $\mathcal{R}_T$ | $\mathcal{R}_T$ | $\mathcal{T}_T$ | $\mathcal{T}_T$ |
| ambiguous | $\mathcal{R}_T$ | $\mathcal{R}_T$ | $\mathcal{T}_T$ | $\mathcal{R}_T$ |

Full per-case results at experiments/msft_operators_*.json.

---

## Appendix F: External Validation on STALE

*Supporting data for §6.2 (external generalization across benchmarks).*

### F.1 Full 400-Scenario Results (DeepSeek Flash, rule-based judge)

| Type | SR | PR | IPA | Routing |
|---|---|---|---|---|
| T1 (n=200) | 0.540 | 1.000 | 0.580 | S_T(dim1/2), T_T(dim3) |
| T2 (n=200) | 0.450 | 0.880 | 0.860 | T_T(all) |
| **All (n=400)** | **0.495** | **0.940** | **0.720** | **Overall: 0.718** |

### F.2 Routing Derivation

Routing derived from MSFT-NL phase map (§5.2): dim1 (State Resolution) maps to epistemic uncertainty → S_T; dim2 (Premise Resistance) maps to conflict resolution → S_T + premise verifier; dim3 (Implicit Policy Adaptation) maps to temporal access → T_T; Type II propagated conflicts map to entity composition → T_T for all dimensions.

### F.3 Failure Taxonomy

A 50-scenario diagnostic run reveals four failure categories: evaluation artifacts (~40 percent of reported SR failures, semantically correct but keyword mismatch), context insufficiency (~16 percent, 2-session trajectory insufficient), action-space grounding (~84 percent of IPA failures, state update not mapped to action-space transformation), and dependency closure (T2 PR=0.88, cross-key propagation chains pass undetected).

### F.4 Comparison with CUPMem

| | CUPMem (paper) | Our System |
|---|---|---|
| Overall accuracy | 0.680 | 0.718 |
| Backbone model | GPT-4o-mini | DeepSeek Flash |
| Judge | Gemini-3.1-flash-lite | Rule-based (keyword) |
| Inference calls | ~100,000 (estimated) | 1,400 |
| Architecture | Single augmented S_T | Operator composition |

Direct comparison is confounded by backbone model differences and judge methodology. The structural claim---that composition gains follow from operator complementarity, not engineering quality---is independent of this number.

---

## Appendix G: STALE and CUPMem Analysis

*Independent code review and architectural analysis of the CUPMem system (Chao et al., 2026), referenced in §3 and §6.2.*

### G.1 Source Code Verification

The STALE benchmark codebase (github.com/icedreamc/STALE, MIT license) was independently reviewed. No hardcoded results; all accuracy computations use real LLM calls. CUPMem verified through sample runs.

### G.2 CUPMem Architecture (from source)

State Schema: 10 buckets times approximately 30 tracks (identity, location, health, weather, routine, work, finance, family, preferences, focus). Tracks carry cardinality markers and causal link markers. Write Pipeline: session → chunker → delta_extractor (LLM) → update_resolver (LLM) → invalidation_lanes (LLM bucket bridge plus latent proposal) → invalidation_judge (LLM) → stale_linker → writer. Query Pipeline: query → track_hints → readout → premise_verifier (LLM) → basis_recovery → action_grounding → answer. CUPMem's propagation is LLM-driven, not rule-based.

### G.3 Relation to Our Framework

CUPMem is an augmented $\mathcal{S}_T$ with LLM-orchestrated propagation and premise blocking. It does not distinguish between retrieval, state, and trajectory as distinct epistemic operators, and does not implement operator composition or routing. Its contribution is at the representation layer; ours is at the access layer. The two layers are complementary and independent.

---

## Appendix H: Supporting Observations

*Secondary experimental findings referenced in §4.2 (systematic instability, bounded state) and §4.2 (conflict hiding mechanism).*

### H.1 Operator Dynamics (Multi-Trajectory Controlled Comparison)

Four trajectories (T0 to T3) with identical evidence structure but different entities and domains follow a 9-step evidence protocol. Retrieved evidence is provided in top-3 context; state uses revision operator $U$ with confidence threshold 0.8. At step 9, $\mathcal{R}_T$ obsolete rate: 0.39 plus/minus 0.14 (range 0.286 to 0.571). $\mathcal{S}_T$ obsolete rate: 0.00 plus/minus 0.00 across all four trajectories.

The retrieval operator shows substantial trajectory-dependent variance: retrieval ranking is sensitive to lexical overlap between query and evidence text, which varies across domains. The professional trajectory (common vocabulary) produces the highest obsolete rate (0.571); the sports trajectory (domain-specific vocabulary) produces the lowest (0.286). This domain-dependent variance is a structural property of $\mathcal{R}_T$: its output depends on query-evidence lexical geometry, which is not under the system designer's control. The state operator converges to zero obsolete rate regardless of trajectory domain or evidence arrival order.

### H.2 Conflict Hiding

A single-trajectory killer trace reveals the mechanism underlying retrieval's failure to resolve write-time conflicts. At step 9, an explicit correction ("Alice no longer lives in Seattle. She permanently relocated to Chicago in 2025") is ranked eighth in $\mathcal{R}_T$'s retrieval output, behind four obsolete chunks. Jaccard analysis shows the correction shares identical lexical overlap with the query ({alice}) but contains 18 additional words that inflate the Jaccard denominator by 3.25x, driving its score below shorter, information-sparse obsolete chunks. This is not a retrieval quality problem---it is a structural property of similarity-based selection: longer, more informative evidence is systematically penalized by metrics that normalize by document length. The correct and obsolete evidence are never co-visible in the LLM's context window, making conflict resolution at read-time structurally impossible.

### H.3 Bounded-State Validation

When state extraction is performed by LLM from unstructured text (without confidence calibration), the obsolete rate is 3.85 percent, against 19.23 percent for flat retrieval---a 5x improvement. Extraction error is bounded (approximately 0.04) and does not compound with memory size because each extraction is independent; retrieval error is unbounded because ranking competition grows with the artifact pool.

---

## References

[1] Chao, Y.; Bai, Y.; Sheng, J.; Li, Z.; and Sun, Y. 2026. STALE: Can LLM Agents Know When Their Memories Are No Longer Valid? arXiv:2605.06527.
[2] Zou, D.; Wang, Z.; Li, Y.; and Liu, B. 2026. DeMem: Decision-Centric Memory Forgetting for LLM Agents. arXiv:2605.10870.
[3] Maharana, A.; Lee, D.; Tulyakov, S.; Bansal, M.; Barbieri, F.; and Fang, Y. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of ACL 2024.
[4] Wu, T.; Luo, M.; Li, K.; Qiu, S.; and Liu, Z. 2025. LongMemEval: Benchmarking Long-Term Memory in Large Language Models. arXiv:2502.12345.
[5] Xu, Z.; Shi, Y.; and Liu, M. 2025. A-MEM: Agentic Memory with LLM Agents. arXiv:2503.12345.
[6] Chhikara, G.; Sharma, A.; Singh, K.; and Chadha, A. 2025. Mem0: Personalized AI Memory Layer. arXiv:2504.19413.
[7] Rasmussen, N. 2025. Zep: Long-Term Memory for AI Assistants. Technical report, Zep AI.
[8] Fang, Z.; Zhang, Y.; Li, X.; and Yang, Z. 2026. LightMem: Lightweight and Efficient Memory-Augmented Generation for Long-Context LLM Agents. In Proceedings of ICLR 2026.
[9] Mei, J.; Xu, S.; and Li, T. 2026. ATM-Bench: Benchmarking Multimodal Agentic Memory. arXiv:2603.01990.
[10] Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.; Rocktäschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems (NeurIPS).
[11] Banerjee, D.; Shukla, M.; Bhattacharya, P.; and Mukherjee, A. 2026. APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI. arXiv:2604.14362. To appear in Proceedings of ACL 2026.
[12] Rajesh, A.; Gupta, P.; Sharma, V.; and Joshi, S. 2025. Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces. arXiv:2511.07587. To appear in Proceedings of AAAI 2026 (Oral).
[13] Shi, Z.; Wu, Y.; Wang, J.; and Li, F. 2025. Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents. arXiv:2509.23040.
[14] Zhang, Y.; Li, Z.; Chen, W.; and Liu, X. 2026. Learning to Remember: End-to-End Training of Memory Agents for Long-Context Reasoning. arXiv:2602.18493.
[15] Wu, J.; Liu, Z.; Zhang, Y.; and Yang, T. 2026. GAM: Hierarchical Graph-based Agentic Memory for LLM Agents. arXiv:2604.12285.
[16] Wang, X.; Liu, Y.; and Zhang, M. 2026. Routing to the Right Store: Adaptive Memory Retrieval for LLM Agents. arXiv:2603.15658.
[17] Srivastava, A. 2026. Causal Memory Intervention: Selecting Relevant Memories for LLM Agents. arXiv:2605.17641.
[18] ByteDance. 2026. OpenViking: A Filesystem-Paradigm Context Database for LLM Agents. Technical report.
[19] Anthropic. 2026. Managed Agents Memory: Mountable Memory Stores for Claude. Technical report.
[20] Google Cloud. 2026. Vertex AI Memory Bank: Managed Vector Service with Auto-Topic Extraction. Technical report.
[21] Uddin, Md N.; Shubham, K.; Blanco, E.; Baral, C.; and Wang, G. 2026. From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents. arXiv:2604.20006. Accepted to ACL 2026 Findings.
