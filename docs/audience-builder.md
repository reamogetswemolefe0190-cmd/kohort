# Kohort Persistent Audience Builder

The Audience Builder enables defining, generating, and caching reusable groups of synthetic customer personas with grounded demographic and psychographic distributions.

---

## 1. Why Persistent Audiences?

Competitors like Minds market synthetic panels with a ~300x markup per response. In Kohort:
1. **Zero Repeated Token Waste**: Personas are generated once, sampled from explicit distributions (age, income, location, Rogers adopter category, price sensitivity), and saved into persistent JSON panel files.
2. **Deterministic Panel Re-use**: Studies testing pricing, messaging variations, or competitive positioning interview the exact same panel, isolating differences in the offer rather than demographic noise.
3. **Product-Blind Guardrails**: Personas are built blind to the client's proposition, preventing priming and bias.

---

## 2. CLI Usage

### List Available Audiences
```bash
python -m kohort audience list
```
Displays all saved panels in `audiences/` along with pre-configured templates in `audiences/templates/`.

### Inspect an Audience
```bash
python -m kohort audience show sa-tech-founders
```
Outputs the demographic scope, target market, and segment allocation (e.g. Bootstrapped Founders 40%, Scaleup CTOs 35%, Agency Directors 25%).

### Build and Save an Audience Panel
```bash
# Build 50 personas offline for free (reproducible seed)
python -m kohort audience build sa-tech-founders --personas 50 --mock --seed 42

# Build with live multi-model LLM generation
python -m kohort audience build sa-tech-founders --personas 50
```

### Running Concept Tests with a Saved Audience
```bash
python -m kohort run briefs/example_loopa.yaml --audience sa-tech-founders --personas 50 --mock
```

---

## 3. Creating Custom Audience Templates

Create a YAML file in `audiences/templates/<name>.yaml`:

```yaml
name: b2b-procurement-leads
title: Enterprise B2B Procurement & IT Buyers
region: South Africa
category: Enterprise Cloud Services
target_market: Procurement officers and VP of IT Infrastructure
description: Enterprise decision makers evaluating multi-million ZAR software and cloud renewals.
sample_size: 50
criteria:
  - "Responsible for supplier selection or procurement approval"
segments:
  - name: Risk-Averse Compliance Gatekeepers
    description: Focused on regulatory compliance, audit trails, and strict SLAs.
    share: 0.50
    age_min: 35
    age_max: 60
    income_levels: [upper-middle, high]
    key_needs: [audit logs, SOC2, POPIA compliance, local data sovereignty]
    likely_objections: [untested security track record, international hosting]
  - name: Cost-Optimization Directors
    description: Mandated to cut IT spend and renegotiate vendor contracts.
    share: 0.50
    age_min: 32
    age_max: 55
    income_levels: [middle, upper-middle, high]
    key_needs: [transparent volume tiers, multi-year discounts, easy migration]
    likely_objections: [hidden implementation costs, variable currency billing]
```
