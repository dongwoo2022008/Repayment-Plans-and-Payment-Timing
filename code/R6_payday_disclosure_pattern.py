"""Payday disclosure (pay_sched_rx): extraction pattern (Section 3.3, Online Appendix H).

The pattern is applied to the raw repayment-plan text. Raw texts contain personal
information and are not distributed, so this script documents the rule and can be
run only by someone holding the texts. Against the indicator in
data/analysis_v3_public.csv it matches 1,928 of 1,929 loans (165 of 166 flagged
plans, no false positives). Loan B100616-2 lists monthly income by date in a
different form and is kept as disclosing.
"""
import re
PAYDAY = re.compile(r"급여일|월급날|월급일|봉급날|\d{1,2}\s*일\s*(?:에)?\s*(?:수령|입금|지급)")

def payday_disclosure(plan_text: str) -> int:
    return int(bool(PAYDAY.search(plan_text or "")))
