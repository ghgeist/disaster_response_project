# Storm Signal — Legacy Code & AI‑Thrash Cleanup Pass

## Purpose (Narrow and Explicit)

This document exists to guide a **single, bounded cleanup pass** over the Storm Signal repository.

Its goal is **not refactoring**, **not redesign**, and **not optimization**.

Its only purpose is:

> Remove or quarantine legacy code and AI‑generated thrash so the repo becomes *legible, trustworthy, and re‑enterable*.

This pass is a **precondition** for any demo, UX, or narrative work.

---

## What This Pass Is / Is Not

**This is:**

* A pruning and clarity pass
* A trust‑rebuilding step
* A way to make future work cheaper

**This is not:**

* A feature pass
* A performance pass
* An architectural redesign
* A commitment to future directions

If a change requires explanation, debate, or future justification, it does **not** belong in this pass.

---

## Operating Rules (Hard Constraints)

1. **No refactors** — only delete, archive, or clearly fence.
2. **No new abstractions** — do not replace removed code with new machinery.
3. **Bias toward removal** — absence is preferable to dubious presence.
4. **Archive beats rewrite** — if unsure, move it out of the execution path.
5. **One‑pass mindset** — this is not iterative cleanup.

---

## Definitions

### Legacy Code

Code that:

* Exists primarily for historical reasons
* Reflects earlier project goals or constraints
* Is no longer part of the current mental model

### AI‑Thrash

Code that:

* Was generated rapidly by AI
* Is only partially understood
* Adds surface area without confidence
* Cannot be explained cleanly after re‑reading

AI‑thrash is not a moral failure — it is expected during exploration. This pass simply ends that phase.

---

## Keep / Archive / Delete Decision Filter

For each major file, module, or directory, ask **in order**:

1. **Do I understand what this does in under 2 minutes?**

   * If no → archive or delete

2. **Does this currently run in a known, tested path?**

   * If no → archive

3. **Would I confidently modify this six months from now?**

   * If no → archive

4. **Is there another file that does the same job?**

   * If yes → keep one, archive the rest

5. **Would deleting this break something I actively use today?**

   * If no → delete

Only code that survives all five questions stays active.

---

## Archiving Rules

Archived code must:

* Live outside the main execution path (e.g. `_archive/` or `legacy/`)
* Be clearly marked as inactive
* Require deliberate effort to resurrect

Archived code does **not** need to be clean, runnable, or documented.

Its job is memory preservation, not usability.

---

## Expected Outcomes (Concrete)

At the end of this pass:

* The repo has fewer files and directories
* The remaining code is explainable end‑to‑end
* There is exactly **one obvious way** to:

  * run the demo (if applicable)
  * train a model (if applicable)
* Uncertainty is visible as absence, not hidden complexity

---

## Explicit Non‑Goals

This pass does **not** aim to:

* Decide final product scope
* Resolve ML vs product positioning
* Perfect the README
* Improve model performance
* Touch UI/UX

Those happen *after* clarity is restored.

---

## Stop Condition

Stop this cleanup pass when:

* You can skim the repo tree and feel oriented
* No file causes immediate confusion or distrust
* You feel relief rather than obligation

If energy drops or second‑guessing begins, stop.

A partially completed cleanup is better than an over‑extended one.
