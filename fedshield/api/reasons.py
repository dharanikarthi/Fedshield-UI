"""Plain-language reason codes for alerts (fraud-officer facing)."""
import math

PCT = lambda v: f"{v * 100:.0f}%"
NUM = lambda v: f"{v:,.0f}"
INR = lambda v: f"₹{v:,.0f}"
DAYS = lambda v: f"{v:,.0f} days"

# feature -> (label, formatter, sentence when it RAISES risk)
TEMPLATES = {
    "bidir_ratio": ("Two-way counterparties", PCT, "Only {v} of counterparties are two-way (typical {m}): money flows in from one set and out to another, the funnel shape of a mule account."),
    "n_bidir": ("Two-way counterparties (count)", NUM, "{v} two-way counterparties (typical {m})."),
    "n_cp_in": ("Distinct senders", NUM, "Received from {v} distinct senders (typical {m}): fan-in."),
    "n_cp_out": ("Distinct beneficiaries", NUM, "Paid out to {v} distinct beneficiaries (typical {m})."),
    "fan_in_out_ratio": ("Fan-in / fan-out", lambda v: f"{v:.1f}×", "Senders outnumber beneficiaries {v} (typical {m}): many-to-few aggregation."),
    "passthrough_48h": ("Funds drained within 48h", PCT, "{v} of incoming money left within 48 hours (typical {m}): rapid pass-through."),
    "passthrough_sameday": ("Same-day pass-through", PCT, "{v} of credits were debited the same day (typical {m})."),
    "passthrough_ratio": ("Debit/credit balance", PCT, "Outflow matched {v} of inflow (typical {m}): the account keeps almost nothing."),
    "balance_drain": ("Balance drain", PCT, "Balance repeatedly drained toward zero (score {v}, typical {m})."),
    "near_zero_bal_share": ("Near-zero balance", PCT, "Balance was under ₹1,000 after {v} of transactions (typical {m})."),
    "max_gap_days": ("Longest dormancy", DAYS, "Account was silent for {v} before activity resumed (typical {m})."),
    "txn_after_dormancy": ("Activity after dormancy", NUM, "{v} transactions in the 30 days after waking from dormancy (typical {m})."),
    "struct_50k_share": ("Just-below-₹50k amounts", PCT, "{v} of transactions were between ₹45,000 and ₹50,000 (typical {m}): structuring below the PAN threshold."),
    "n_struct_50k": ("Just-below-₹50k count", NUM, "{v} transactions just below ₹50,000 (typical {m})."),
    "round_1k_share": ("Round amounts", PCT, "{v} of amounts were exact multiples of ₹1,000 (typical {m})."),
    "mobile_spike_ratio": ("Post-mobile-change spike", lambda v: f"{v:.1f}×", "Activity rose {v} in the 60 days after the mobile number changed (typical {m}): possible account takeover."),
    "days_since_mobile_update": ("Mobile number changed", DAYS, "Mobile number changed {v} ago (typical {m})."),
    "account_age_days": ("Account age", DAYS, "Account is {v} old (typical {m})."),
    "volume_to_balance": ("Volume vs balance", lambda v: f"{v:.0f}×", "Monthly inflow is {v} the average balance (typical {m}): income mismatch."),
    "n_ip": ("Distinct IPs/devices", NUM, "Used from {v} distinct IP addresses (typical {m})."),
    "geo_spread": ("Geographic spread", lambda v: f"{v:.2f}°", "Transactions spread over {v} of latitude/longitude (typical {m}): geographic anomaly."),
    "n_locations": ("Distinct locations", NUM, "Transacted from {v} distinct locations (typical {m})."),
    "night_share": ("Night activity", PCT, "{v} of transactions happened between 11pm and 5am (typical {m})."),
    "ch_upi_credit": ("UPI credits", PCT, "{v} of transactions were incoming UPI (typical {m})."),
    "ch_atm_wd": ("ATM withdrawals", PCT, "{v} of transactions were ATM cash-outs (typical {m})."),
    "ch_imps": ("IMPS share", PCT, "{v} of transactions used IMPS (typical {m})."),
    "top_beneficiary_share": ("Top-beneficiary concentration", PCT, "{v} of outflow went to a single beneficiary (typical {m})."),
    "txn_velocity": ("Transaction velocity", lambda v: f"{v:.1f}/day", "{v} transactions per day (typical {m})."),
    "burst_ratio": ("Burst intensity", lambda v: f"{v:.1f}×", "Busiest day had {v} the usual daily volume (typical {m})."),
    "reg_cp_rare": ("Shared rare counterparties", lambda v: f"{v:.2f}", "Shares rare counterparties with mules confirmed at consortium banks (score {v}, typical {m})."),
    "reg_cp_flagged": ("Counterparties linked to confirmed mules", NUM, "{v} counterparties also transact with confirmed mules at consortium banks (typical {m})."),
    "reg_cp_share": ("Share of flagged counterparties", PCT, "{v} of counterparties are in the consortium suspect registry (typical {m})."),
    "reg_ip_shared_mule": ("Device shared with a confirmed mule", NUM, "Shares {v} IP addresses/devices with confirmed mule accounts (typical {m})."),
    "kyc_compliant": ("KYC compliant", lambda v: "yes" if v >= 0.5 else "no", "KYC compliant: {v}."),
    "days_since_kyc": ("Days since KYC", DAYS, "Last KYC was {v} ago (typical {m})."),
    "unique_channels": ("Channels", NUM, "{v} channels used (typical {m})."),
    "n_channels": ("Channels used", NUM, "Used {v} payment channels (typical {m})."),
    "n_txn": ("Transactions", NUM, "{v} transactions in the window (typical {m})."),
    "sum_credit": ("Total credits", INR, "{v} received in the window (typical {m})."),
}


def _fmt(fmt, v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return fmt(v)
    except Exception:
        return str(v)


def explain(features, values, contribs, medians, k=5):
    """Top-k risk-raising reasons, plus the strongest mitigating factor."""
    order = sorted(range(len(features)), key=lambda i: -contribs[i])
    out = []
    for i in order:
        if contribs[i] <= 0 or len(out) >= k:
            break
        f = features[i]
        m = medians.get(f)
        if m is not None and values[i] is not None and abs(float(values[i]) - float(m)) <= 1e-9 * max(1.0, abs(float(m))):
            continue          # value is typical: the effect is an interaction, not a sentence an officer can use
        label, fmt, sent = TEMPLATES.get(f, (f.replace("_", " "), lambda v: f"{v:,.3g}", "{label}: {v} (typical {m})."))
        out.append({"feature": f, "label": label, "value": values[i], "typical": medians.get(f),
                    "impact": float(contribs[i]),
                    "text": sent.format(v=_fmt(fmt, values[i]), m=_fmt(fmt, medians.get(f)), label=label)})
    neg = min(range(len(features)), key=lambda i: contribs[i])
    mitig = None
    if contribs[neg] < 0:
        f = features[neg]
        label, fmt, _ = TEMPLATES.get(f, (f.replace("_", " "), lambda v: f"{v:,.3g}", ""))
        mitig = {"feature": f, "label": label, "impact": float(contribs[neg]),
                 "text": f"Mitigating: {label} = {_fmt(fmt, values[neg])} (typical {_fmt(fmt, medians.get(f))})."}
    return out, mitig
