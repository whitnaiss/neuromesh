# Yeo 7x7 Connectivity Source — Week 1 Findings

**Track A (Mihir Mathur) · Week 1 deliverable: "Locate the Yeo 7x7 connectivity source"**
**Status: source located and verified; the 7x7 matrix itself must be DERIVED (not downloaded).**
**No numerical connectivity values are invented anywhere in this repository.**

This document records what was verified during Week 1, what remains unresolved,
and the reproducible procedure for producing the matrix in Week 2. The
Yeo-SBM generator is explicitly out of Week 1 scope and is **not** implemented.

---

## 1. The four things that must not be conflated

The Week 1 brief asks for these to be kept distinct. They are genuinely
different objects, and conflating them is the main correctness risk here.

| # | Object | What it is | Is it a 7x7 matrix? |
|---|--------|------------|---------------------|
| 1 | **Yeo 7-network functional organization** | A *parcellation / labelling*: a cortical clustering that assigns every vertex to one of 7 networks. It is a partition, not a connectivity matrix. | **No.** It carries no edge weights at all. |
| 2 | **Structural connectivity (SC)** | Physical white-matter connectivity, measured with diffusion MRI tractography. Typically streamline counts or densities between parcels. | No — it is parcel x parcel (e.g. 400x400). |
| 3 | **Functional connectivity (FC)** | Statistical dependence between regional BOLD time series at rest (usually Pearson/Fisher-z correlation). Not a physical connection. | No — also parcel x parcel. |
| 4 | **Derived 7x7 network-level matrix** | What the Yeo-SBM generator needs: inter-network block connectivity, obtained by **aggregating** (2) or (3) over the labels in (1). | **Yes — but it is computed by us.** |

The project abstract commits specifically to structural connectivity:
> "a Yeo-based stochastic block model (Yeo-SBM) whose between-network
> connectivity is **informed by human structural connectivity**."

So object (2) aggregated over object (1) is the intended path, and FC is at
most a secondary/robustness comparison. This matters: SC and FC give materially
different 7x7 matrices (FC has strong within-network blocks by construction,
since the Yeo networks were *defined* by clustering FC — using FC would be
partly circular).

---

## 2. Verified sources

All four were checked during Week 1. Accession details are given so a
teammate can confirm them independently.

### 2.1 The parcellation (object 1) — VERIFIED

**Yeo, B. T. T., Krienen, F. M., Sepulcre, J., et al. (2011). "The organization
of the human cerebral cortex estimated by intrinsic functional connectivity."
*Journal of Neurophysiology* 106(3), 1125–1165. DOI: 10.1152/jn.00338.2011**

- Method: surface-based alignment of resting-state fMRI from 1,000 subjects,
  then clustering to identify and replicate 7- and 17-network solutions.
- Distribution: FreeSurfer, in `fsaverage` surface space and nonlinear MNI152
  space — FreeSurfer wiki page `CorticalParcellation_Yeo2011`.
  Also mirrored in NeuroVault collection 1057.
- The seven networks: Visual, Somatomotor, Dorsal Attention, Ventral Attention
  (Salience), Limbic, Frontoparietal (Control), Default.
  *(The Frontoparietal network is the one the abstract designates as the
  Distributed Aggregation readout group.)*

### 2.2 A finer parcellation already mapped to the 7 networks — VERIFIED

**Schaefer, A., Kong, R., Gordon, E. M., et al. (2018). "Local-Global
Parcellation of the Human Cerebral Cortex from Intrinsic Functional
Connectivity MRI." *Cerebral Cortex* 28(9), 3095–3114.**

- Repository: `github.com/ThomasYeoLab/CBIG`, under
  `stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/`.
- **Verified present** in `Parcellations/MNI/`: files named
  `Schaefer2018_{100..900}Parcels_7Networks_order_FSLMNI152_{1,2}mm.nii.gz`.
- The `*_7Networks_order` naming is the key asset: **each parcel's name encodes
  its Yeo network**, e.g. `7Networks_LH_Vis_1`, `7Networks_RH_Cont_Par_1`.
  This is the label vector needed to aggregate a parcel-level matrix to 7x7.
- The label lookup tables live in the `freeview_lut/` and `fsleyes_lut/`
  subdirectories (e.g. `Schaefer2018_400Parcels_7Networks_order.txt`).
  **Not yet downloaded** — see §5.
- Licence: CBIG repository licence — **must be checked before redistributing
  any derived file.** Not yet reviewed.

### 2.3 Group-level structural AND functional connectivity matrices — VERIFIED

**ENIGMA Toolbox** — `enigma-toolbox.readthedocs.io`, "Connectivity data" page.
Preprint DOI: **10.1101/2020.12.21.423838**.

This is the most practical source found: it ships **pre-computed group-average
HCP matrices in exactly the Schaefer parcellations above**, so no raw
diffusion data or tractography pipeline is needed.

Verified details, quoted/paraphrased from the documentation page:

- Cohort: "a cohort of unrelated healthy adults from the Human Connectome
  Project (HCP)". **The exact subject count is NOT stated on that page** — see
  §5, unresolved item U1. *Do not cite a number for this until it is read out
  of the preprint.*
- Parcellations available: Desikan-Killiany, Glasser, and **Schaefer 100, 200,
  300, 400**.
- **Structural** connectivity: MRtrix3, anatomically-constrained tractography;
  initial tractogram of 40 million streamlines; maximum tract length 250;
  FA cutoff 0.06; SIFT2 applied. "Structural connectivity was defined by the
  number of streamlines between two regions." The group-average matrix uses
  distance-dependent thresholding and a log transform.
- **Functional** connectivity: pairwise correlations between regional time
  series; "negative connections were set to zero"; subject matrices
  z-transformed and averaged into a group-average connectome; the released
  matrices are unthresholded.
- Python API: `from enigmatoolbox.datasets import load_sc, load_fc`, each
  returning `(cortical_matrix, cortical_labels, subcortical_matrix,
  subcortical_labels)`.

**Caveat to carry into the write-up:** the group-average SC matrix is already
log-transformed and distance-thresholded. Those are non-linear, non-invertible
transforms, so block means computed from it are means *of transformed
streamline counts*, not of raw counts. This must be stated in the methods
section, and it is a reason to also report the FC-derived matrix as a
sensitivity check.

### 2.4 Alternative fetchers — VERIFIED TO EXIST, not yet tested

- `netneurotools.datasets.fetch_schaefer2018` — fetches the Schaefer
  parcellation with its 7-network ordering.
- `nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=..., yeo_networks=7,
  resolution_mm=...)` — a `yeo_networks` parameter is documented by
  third-party wrappers; **the exact nilearn signature for the installed
  version was not confirmed in Week 1** (unresolved item U2).
- `nilearn.datasets.fetch_atlas_yeo_2011` — exists, but at least one older
  nilearn release had a broken download URL requiring a manual `url=`
  argument. Verify before relying on it.

---

## 3. The central negative finding

**No published, citable, canonical 7x7 Yeo-network connectivity matrix was
located.**

Searches for a published network-level 7x7 inter-network structural matrix
returned only parcel-level matrices and papers that *order* their parcel-level
matrices by the seven networks for visualisation. The closest hits were
functional, not structural, and none provided a reusable 7x7 table.

Consequences, and the reason this is reported rather than worked around:

1. The 7x7 matrix **must be derived by us** from a parcel-level matrix plus the
   Yeo label vector. It is a computation we own, document and test — not a
   download.
2. Because it is derived, the aggregation rule is a **methodological choice we
   must state explicitly** (see §4, step 5). Different defensible rules give
   different matrices.
3. **Nothing in this repository may contain hand-written 7x7 numbers.** If
   Week 2 cannot obtain the data, Yeo-SBM must be reported as blocked, not
   populated with plausible-looking values.

---

## 4. Reproducible derivation procedure (for Week 2)

Target artifact: `data/yeo/yeo7x7_sc.json` — a 7x7 matrix of inter-network
connection *densities* plus full provenance. Not yet created.

**Required format** (proposed, so Week 2 starts from a fixed contract):

```json
{
  "schema_version": 1,
  "networks": ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
  "matrix": [[...7 floats...], "... 7 rows, symmetric ..."],
  "diagonal_is": "within-network density",
  "provenance": {
    "source": "ENIGMA Toolbox load_sc()",
    "source_version": "<package version>",
    "parcellation": "Schaefer2018_400Parcels_7Networks",
    "modality": "structural",
    "aggregation": "mean of log-transformed streamline counts over parcel pairs in each block",
    "normalisation": "<rule>",
    "retrieved_utc": "<timestamp>",
    "citation": ["Yeo 2011 10.1152/jn.00338.2011", "Schaefer 2018", "ENIGMA 10.1101/2020.12.21.423838"]
  }
}
```

**Steps:**

1. `pip install enigmatoolbox` (check licence first). Record the version.
2. `sc, sc_labels, _, _ = load_sc(parcellation='schaefer_400')` — confirm the
   returned matrix is 400x400 and symmetric, and that `sc_labels` has 400
   entries.
3. Parse each label in `sc_labels` to its Yeo network via the
   `7Networks_<hemi>_<Network>_<...>` naming, producing a length-400 label
   vector over the 7 names. **Assert that all 7 networks are present and that
   no parcel is unassigned** — a silent parsing miss here would corrupt every
   downstream block.
4. Verify the label vector against an independent source
   (`netneurotools.datasets.fetch_schaefer2018`) before trusting it.
5. Aggregate to 7x7. **State the rule explicitly.** The recommended default:
   for block (A, B), the **mean edge weight over all parcel pairs (i in A,
   j in B)**, excluding the diagonal for A == B. Mean (not sum) is required
   because the seven networks contain unequal parcel counts, so sums would
   simply re-encode network size.
6. Convert the 7x7 weights to SBM *probabilities* by rescaling so the expected
   mean degree of the generated graph equals 4 at each N — this is the
   "degree rescaling" the Week 2 plan already names. The relative block
   structure comes from the data; the overall density is set by our matched
   mean-degree constraint.
7. Record provenance, write the JSON, and add a test asserting symmetry,
   7x7 shape, non-negativity, and that all 7 network names are present.

**Boundary condition to handle in Week 2:** balanced role assignment needs
N/7 agents per network, which is exact for N = 7, 14, 28, 56, 112 (all
divisible by 7). At N = 7 each network has exactly one agent, so the
Yeo-SBM's within-network block has no edges to place and the graph is
determined entirely by the between-network blocks. Expect N = 7 Yeo-SBM to be
degenerate, consistent with N = 7 already being a declared boundary case.

---

## 5. Unresolved items (carry into Week 2)

| ID | Item | Why it matters | Action |
|----|------|----------------|--------|
| U1 | ENIGMA HCP **subject count** not stated on the docs page | Needed for the methods section; must not be guessed | Read preprint 10.1101/2020.12.21.423838 |
| U2 | Exact `nilearn.datasets.fetch_atlas_schaefer_2018` signature for the installed version | Affects whether nilearn is a usable fallback | Check installed nilearn's API reference |
| U3 | CBIG and ENIGMA **licence terms** | Governs whether a derived matrix can be committed to a public repo | Read both licences before committing `yeo7x7_sc.json` |
| U4 | Schaefer parcel-count choice (100 / 200 / 400) | Coarser parcellations give noisier block means | Decide in Week 2; 400 is the default proposal |
| U5 | Whether to use SC (primary) or also FC (sensitivity) | FC is partly circular with the Yeo clustering | Primary = SC per the abstract; report FC as a check |
| U6 | None of the packages in §2.3–2.4 are installed or executed yet | The procedure in §4 is verified-by-documentation, not by running it | Install and run in Week 2 |

---

## 6. If the data cannot be obtained

Fallback, in preference order — all to be **reported as a deviation**, never
presented as the data-informed version:

1. A different public group-average connectome in a Yeo-labelled parcellation.
2. A published figure's reported block values, cited, with the extraction
   method stated — weaker, and only if numerically reported in text/tables,
   never read off a colour map.
3. Report Yeo-SBM as **blocked** and run the study with the four remaining
   topologies, stating the omission in the report.

Option 3 is strongly preferred over inventing numbers.

---

## References

1. Yeo, B. T. T., et al. (2011). *J Neurophysiol* 106(3), 1125–1165. DOI: 10.1152/jn.00338.2011
2. Schaefer, A., et al. (2018). *Cerebral Cortex* 28(9), 3095–3114. CBIG: `github.com/ThomasYeoLab/CBIG`
3. ENIGMA TOOLBOX. Preprint DOI: 10.1101/2020.12.21.423838 · Docs: `enigma-toolbox.readthedocs.io`
   *(Author list not yet read from the preprint — retrieve the full citation
   together with unresolved item U1 before using it in the report.)*
4. FreeSurfer wiki, `CorticalParcellation_Yeo2011` · NeuroVault collection 1057
