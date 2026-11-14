# Repayment Plans and Payment Timing among Financially Excluded Borrowers: Evidence from Online Lending

Replication materials for the manuscript: the processed data, the annotation codebooks, the validation label files, and the code that produces the tables, figures, and appendix results.

## What is and is not distributed

The repayment plans and other application texts written by borrowers contain personal and financial information and are **not** distributed. Neither are the raw platform records. The files here contain the variables derived from the texts (model annotations and rule-based indicators) and the installment records, keyed by platform listing codes that carry no names. Free-text evidence notes produced during annotation are not included.

Three items are available from the author on request, subject to data-protection requirements: the raw texts, the Phase B human coding and adjudication records, and the Phase B sampling seed. Results that depend on them are marked below.

## Contents

```
data/
  analysis_v3_public.csv   Loan-level analysis file (1,929 loans, 80 columns).
  panel_long.csv           Installment panel (19,755 rows): scheduled and actual dates,
                           status, delay, amounts.
  listing_history.csv      Repayment history at listing (1,929 loans), built from the
                           panel by code/R5a_listing_time_history.py.
  loan_level_pcg.csv       Loan-level file for the plan-capacity gap (Online Appendix F).
  annot_v3.csv             Model annotation under the frozen codebook (labels only).
codebooks/
  CODEBOOK_v2.md           First codebook. English translation and Korean original.
  CODEBOOK_v3.md           Frozen codebook used for the timing measures. English
                           translation and Korean original (the prompt).
validation/
  llm_key.csv              Full-corpus model annotation (labels only).
  llm_rerun_2026-08.csv    Re-annotation of the 120 Phase B cases.
  pb_map_final.csv         Phase B case-to-loan mapping.
  phaseA_human_g3.csv      Phase A human coding (60 cases).
  phaseA_g3_merged.csv     Phase A human coding merged with model labels.
code/                      See the table below.
figures/
  Figure_1.png, Figure_2.png, Figure_G1.png
```

`annot_v3.csv`, `validation/llm_key.csv`, and `loan_level_pcg.csv` contain 1,930 rows: they also cover loan B111107-14, which was annotated but is not in the analytic sample. The scripts restrict to the 1,929 analytic loans.

## Code and the results it produces

Run every script from the repository root, for example `python code/R1b_contract_anchored_benchmarks_gap.py`.

| Script | Produces |
|---|---|
| `R1b_contract_anchored_benchmarks_gap.py` | Table 3, Panel A: observed shares and Benchmark 1; Online Appendix H |
| `R1c_contract_anchored_benchmarks_calendar.py` | Table 3, Panels A and B: Benchmark 2, payday-only loans, window sensitivity |
| `R1d_benchmark_calibration_and_timing.py` | Online Appendix H: benchmark calibration; timing of payments near the stated day |
| `R7_sibling_loan_alignment.py` | Table 3, Panel C; Section 5.1 and Online Appendix H (sibling loans) |
| `R7b_sibling_calibration.py` | Online Appendix H: calibration of the sibling-loan test |
| `R5a_listing_time_history.py` | `data/listing_history.csv` |
| `R5b_delinquency_models.py` | Tables 5 and 6; Section 5.3; Online Appendices G and H |
| `R5c_repayment_success.py` | Table 4; Section 5.2; Online Appendices D, F, G, and H |
| `validation_agreement.py` | Online Appendix E: run-to-run stability of the annotation |
| `make_fig1_alignment_dueday.py` | Figure 1 (pass the output path as the first argument) |
| `make_fig2_trajectories.py` | Figure 2 (output path as the first argument) |
| `make_figG1_severity.py` | Figure G.1 |
| `R5_lib.py` | Shared helpers (data loading, complementary log-log estimator) |

Four scripts document rules or analyses that were applied to the raw texts and cannot be run from the public files:

| Script | Documents |
|---|---|
| `R6_payday_disclosure_pattern.py` | The text pattern behind the payday-disclosure indicator (`pay_sched_rx`) |
| `salary_day_pattern.py` | The text pattern behind the day of a payday (`salday`) |
| `style_features.py` | The two writing-style proxies (`S2`, `S3`; Online Appendix D) |
| `R1e_numeral_stated_day_subsample.py` | Alignment in the 87 loans whose plan text contains the stated day as a numeral (Online Appendix H) |

Permutation and bootstrap results are Monte Carlo. Seeds are set in the scripts.

Not reproducible from this repository: the Phase B agreement statistics between human coders and between the model and the adjudicated labels (Table 2; Online Appendix E), which rest on the coding records held by the author.

## Notes on the data

Categorical values are kept in Korean as recorded by the platform, and the scripts refer to them directly.

- `outcome`: 완납 (fully repaid), 손실 (charged off), 진행중 (ongoing)
- loan `status`: 정상종료 (closed, repaid), 연체중 (in delinquency), 상환중 (repaying), 손실(상환완료) (charged off, settled), 상환지연중 (repayment delayed)
- panel `status`: 일반상환 (regular payment), 일시상환 (lump-sum payment), 중도상환 (early payoff), 연체 (delinquent), 손실 (charged off)
- `gender`: 남자 (male), 여자 (female); `freq`: 월 (monthly), 일시 (lump sum)
- `age`: years followed by 세; a bare 세 means not reported
- `job`: 25 occupation categories

Key variables in `analysis_v3_public.csv`: `day_v3` (stated repayment day), `pay_sched_rx` (payday disclosure), `salday` (day of a payday named in the plan), `P2` (funding-source specificity), `P3v3` and `P3b` (repayment timing, scale and indicator), `bgroup` (borrower identifier), `S2` and `S3` (writing-style proxies).

## Environment

Python 3.10 or later with `numpy`, `pandas`, and `scipy`. The figure scripts also need `matplotlib` and `Pillow`.

## License

The code in `code/` is released under the MIT License (see `LICENSE`). The data, codebooks, validation files, and figures are released under the Creative Commons Attribution 4.0 International License (see `LICENSE-DATA`). Please cite the manuscript when using these materials.
