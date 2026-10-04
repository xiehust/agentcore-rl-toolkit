# GPU Spot Pricing, Availability & Quota Survey

**Account:** <ACCOUNT_ID> · **Surveyed:** 2026-09-23 15:39 UTC · **Method:** read-only AWS CLI
**Regions:** us-west-2 (primary), us-east-1, us-east-2

---

## (a) Per region / type: AZ availability, latest spot price, on-demand price

On-demand price is uniform across all three regions (per-hour, Linux, shared tenancy):
p5.4xlarge **$6.88**, p5.48xlarge **$55.04**, p5en.48xlarge **$63.296**.

### us-west-2
| Type | AZs offering | Spot range ($/hr) | Latest spot | On-demand | Spot discount |
|---|---|---|---|---|---|
| p5.4xlarge | 2a, 2b, 2c, 2d | 2.6295 (flat) | 2.6295 | 6.88 | ~62% |
| p5.48xlarge | 2a, 2b, 2c, 2d | 20.81–21.04 | 21.04 (2b) | 55.04 | ~62% |
| p5en.48xlarge | 2a, 2c, 2d | 27.01–27.28 | 27.28 (2a) | 63.296 | ~57% |
| p5e.48xlarge | 2c only | (informational) | — | — | — |

### us-east-1
| Type | AZs offering | Spot range ($/hr) | Latest spot | On-demand | Spot discount |
|---|---|---|---|---|---|
| p5.4xlarge | 1a, 1b, 1c, 1d, 1e, 1f | 2.6024–2.6295 | 2.6024 (1b) | 6.88 | ~62% |
| p5.48xlarge | 1a, 1b, 1c, 1d, 1e, 1f | 20.31–21.04 | 21.04 (1b/1d) | 55.04 | ~62% |
| p5en.48xlarge | 1a, 1c | 27.07–27.37 | 27.07 (1a) | 63.296 | ~57% |

### us-east-2
| Type | AZs offering | Spot range ($/hr) | Latest spot | On-demand | Spot discount |
|---|---|---|---|---|---|
| p5.4xlarge | 2a, 2b, 2c | **2.5176–2.6295** | **2.5221 (2b)** | 6.88 | **~63%** |
| p5.48xlarge | 2a, 2b, 2c | **20.26–20.51** | 20.42 (2a) | 55.04 | ~63% |
| p5en.48xlarge | 2a, 2b, 2c | 27.15–27.37 | 27.15 (2a) | 63.296 | ~57% |

**Cheapest spot per type (all regions):** p5.4xlarge → **us-east-2b $2.5221**; p5.48xlarge → **us-east-2c $20.257**; p5en.48xlarge → us-east-1a $27.066.

---

## (b) Quotas per region + verdict

Quota **L-3819A6DF** = "All P Spot Instance Requests" (measured in **vCPUs**).
Quota **L-417A185B** = "Running On-Demand P instances" (vCPUs, informational).

| Region | P Spot vCPU (L-3819A6DF) | On-Demand P vCPU (L-417A185B) | Enough for p5.4xlarge (16 vCPU)? | Enough for 48xlarge (192 vCPU)? |
|---|---|---|---|---|
| us-west-2 | **64** | 768 | ✅ yes (up to 4 concurrent) | ❌ **NO** (192 > 64) |
| us-east-1 | **64** | 768 | ✅ yes (up to 4 concurrent) | ❌ **NO** (192 > 64) |
| us-east-2 | **64** | 384 | ✅ yes (up to 4 concurrent) | ❌ **NO** (192 > 64) |

**Verdict:** Spot quota is **64 vCPU in every region** → supports **p5.4xlarge (1× H100)** comfortably (4 in parallel), but **cannot launch any 8-GPU 48xlarge on spot** (needs 192 vCPU). A quota increase request would be required for the large boxes.

---

## (c) Instance specs

| Type | vCPU | RAM | GPU | GPU mem/card | Local NVMe |
|---|---|---|---|---|---|
| p5.4xlarge | 16 | 256 GiB | 1× H100 | 80 GB (81920 MiB) | 3,800 GB |
| p5.48xlarge | 192 | 2,048 GiB | 8× H100 | 80 GB | 30,400 GB |
| p5en.48xlarge | 192 | 2,048 GiB | 8× H200 | 141 GB (144384 MiB) | 30,400 GB |

---

## (d) Infra checks (us-west-2)

- **Default VPC:** ✅ `vpc-0edf3a4e323c23b22` (CIDR 172.31.0.0/16)
- **Key pairs:** `4344-us-west-2`, `lab-key-pair`
- **Deep Learning Base OSS Nvidia Driver GPU AMIs (latest, 20260922):**
  - Ubuntu 24.04: **`ami-07d69ce07bfe5628f`** — supports P4d/P4de/P5/P5e/P5en/P6-B200/P6-B300
  - Ubuntu 22.04: **`ami-07e48b17b73736ba3`**
  - NVIDIA driver version not exposed in AMI name/description (see AWS release notes for the exact driver build).

---

## (e) Recommendation

**Cheapest viable single instance:** **p5.4xlarge (1× H100 80GB) spot in us-east-2, AZ us-east-2b @ ~$2.52/hr.**
This is within the 64-vCPU spot quota (16 vCPU) and is the lowest observed price of any bookable GPU box.

If the workload must stay in the primary region: **p5.4xlarge spot in us-west-2 @ $2.6295/hr** (any AZ 2a–2d), quota OK.

**Estimated cost — 24 GPU-instance-hours (single p5.4xlarge for 24h, or 24 instance-hours total):**
| Region / AZ | Spot $/hr | 24 h |
|---|---|---|
| us-east-2b (cheapest) | 2.5221 | **~$60.53** |
| us-west-2 (primary) | 2.6295 | **~$63.11** |
| On-demand (any region) | 6.88 | ~$165.12 |

Spot saves **~62–63%** vs on-demand. The 8-GPU 48xlarge boxes are **blocked by the 64-vCPU spot quota** and would need a Service Quotas increase before use.

---

## Command status
All commands succeeded (read-only). Notes:
- `get-service-quota` succeeded in all regions (no fallback to `get-aws-default-service-quota` needed).
- On-demand p5en.48xlarge / p5.4xlarge us-east-1 initially returned a $0 `capacitystatus` variant; the real `Used` price was confirmed on retry (p5en.48xlarge = $63.296, p5.4xlarge = $6.88).
- p5.4xlarge is offered in all three regions (contrary to the caveat).
