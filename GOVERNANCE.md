# Governance

This document describes how decisions are made in NeuroFlow. It is intentionally
short: at Stage 1/2 we are a small project, and elaborate governance is a
liability, not an asset. The document grows as the project grows.

## Roles

- **Lead maintainer (BDFL):** NeuroFlow Contributors. Has final say on
  architectural decisions, IR format changes, and release timing.
- **Maintainers:** Trusted contributors with write access. Can merge PRs
  that pass CI, cut releases, and triage issues. Currently 1.
- **Contributors:** Anyone who opens a PR or issue. You become a maintainer
  by sustained, high-quality contributions over time (typically 6+ months
  of merged work).

## How decisions are made

| Type of decision | Who decides | How |
|---|---|---|
| Bug fix, test, doc, CI tweak | Any maintainer | PR with green CI |
| New operator (backward compatible) | Maintainers | PR + 1 approving review |
| NeuroIR version bump (breaking) | Lead maintainer | Public RFC, 7-day comment window |
| C++ ABI / API change | Lead maintainer | Public RFC, 7-day comment window |
| Governance change | Lead maintainer | PR to this file, 7-day comment window |
| Release tagging | Lead maintainer | Manual, after CI green |

The "7-day comment window" rule is intentionally short — we move fast —
but the comment window is real: a -1 from a maintainer blocks the change
until the objection is resolved or overruled by the lead.

## Becoming a maintainer

Nomination by an existing maintainer, no objections from other maintainers
during a 7-day window, then write access is granted. Self-nomination is
fine if you have 6+ months of contributions.

## Removing a maintainer

A maintainer is considered inactive after 6 months of no merged
contributions and no issue / PR triage activity. Inactive maintainers
are moved to "emeritus" status; their write access is removed but their
author credit is preserved. A maintainer can also step down voluntarily
by notifying the lead.

A maintainer can be removed for cause (Code of Conduct violation, malicious
activity) by unanimous vote of the remaining maintainers.

## Conflict of interest

The lead maintainer recuses from any decision in which they have a direct
material interest (e.g. employer-related). The remaining maintainers make
the call in that case.

## License

NeuroFlow is licensed under Apache 2.0. Governance changes do not affect
existing licenses of contributed code.
