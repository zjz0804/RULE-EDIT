# RULE-EDIT

Dataset for **RULE-EDIT: Benchmarking Rule-Level Knowledge Editing in Large Language Models** (EMNLP 2026).

RULE-EDIT evaluates whether knowledge edits propagate between mathematical formulas, natural-language descriptions and numerical instances. It contains **79 geometric rules** covering **15 shapes**, with **44 rules for 2D figures** and **35 for 3D solids**.

`target_new` specifies an intentional counterfactual formula, description or numerical answer. `ground_truth` provides the corresponding reference under ordinary Euclidean geometry.

| Dataset | Cases | Contents |
| --- | ---: | --- |
| Formula editing | 79 | One formula edit, Description RU and four IP instances per rule |
| Description editing | 79 | One description edit, Formula RU and four IP instances per rule |
| Instance editing | 79 | Two sequential edits, two further IP instances and both RU views |
| Instance ICL | 79 | One demonstration followed by a query for a second instance |
| Hierarchical editing | 5 | Symbolic and numerical queries for 11 downstream dependencies, 22 questions in total |

Formula and Description editing share **316 numerical instances**. Instance editing uses a separate set of **316 instances**: 158 for editing and 158 for IP evaluation. All eight assignments within a rule are distinct across these sets, including applicable symmetric side/dimension permutations.

## Files

```text
data/
  prepare_data.py              # Data generation and validation
  raw/
    rules.json                 # Structured rule definitions
    hierarchy.json             # Hierarchical dependencies
  processed/
    formula_editing.json
    description_editing.json
    instance_editing.json
    instance_editing_icl.json
    hierarchical.json
    manifest.json              # Version, seed, hashes, assignments and geometry witnesses
tests/
  test_data_pipeline.py
```

## Data fields

| Field | Meaning |
| --- | --- |
| `case_id` | Main-task index, 0–78; hierarchical cases use a separate 0–4 namespace |
| `subject`, `rule` | Shape and rule name |
| `prompt`, `target_new` | Editing query and counterfactual target |
| `ground_truth` | Pre-edit mathematical references |
| `rephrase_prompt` | A differently worded query with the same inputs |
| `Rule_Understanding` | Cross-view prompts and targets; Formula then Description for instance editing |
| `Instance` | Numerical IP queries, counterfactual targets and pre-edit references |
| `locality` | Two `neighborhood` queries for other rules of the same shape and two `distracting` queries for other shapes |
| `hierarchical` | Downstream symbolic and numerical queries with both reference types |

Formula and Description editing use scalar editing strings. Instance editing uses lists of length two. The ICL variant includes its demonstration in both the main and paraphrased queries. Source definitions use semantic `rule_id` identifiers; processed files use consecutive `case_id` indices.

## Generation

Python **3.9+** is sufficient. The pipeline uses only the standard library and runs locally.

```bash
python data/prepare_data.py build
```

The default seed is **42**. Generation checks geometric feasibility, exact counterfactual answers, cross-view consistency, locality references and separation of numerical samples before writing outputs. Existing files with different contents require `--overwrite`.

To generate another sample:

```bash
python data/prepare_data.py build --seed 43 --output-dir generated/sample_43
```

Primitive scalar inputs are drawn from 1–15. Interior angles are 1–3 radians, and sector angles are 1–6 radians. Derived areas and perimeters may exceed these ranges. Dependent measurements are generated consistently, with constructive witnesses recorded in the manifest where applicable. Assignments for which the counterfactual and reference numerical answers coincide are excluded.

Formula and Description targets are generated from the same expression. Counterfactual numerical answers use exact rational arithmetic and symbolic powers of π. Irrational reference answers include a symbolic substitution and a 12-significant-digit approximation. Hierarchical targets compose the edited rule through the specified dependencies.

## Verification

To verify saved files against a deterministic rebuild:

```bash
python data/prepare_data.py validate
```

This optional command performs the generation checks and compares all output files byte for byte. For a different seed or directory, pass the same options used during generation. To run the automated tests:

```bash
python -m unittest discover -s tests -v
```
