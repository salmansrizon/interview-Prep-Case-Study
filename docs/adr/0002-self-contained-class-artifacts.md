# 0002. Every class's Lecture and Project are self-contained

## Status
Accepted

## Context
Week 8 (CNNs) is conceptually one continuous project: Class 15 builds a baseline MNIST CNN, Class 16 optimizes the same model (dropout, batch norm, early stopping). The natural instinct is to make Class 16's Project a thin diff on top of Class 15's — less duplication, one canonical codebase. But the bootcamp is taught as independent live sessions (`Course Module.pdf`'s Class N units), and a support/assignment session for Class 16 needs to run standalone without first having Class 15's folder open. This tension will recur any time two consecutive classes share an underlying concept (e.g., future weeks that iterate on one model or app across two sessions).

## Decision
Every `Class N Project` folder is a full, independently runnable copy — not a diff or an import from another class's folder — even when it conceptually extends a prior class's work. Week 8 Class 2 Project is a full copy of Week 8 Class 1 Project's scaffold with the optimization techniques layered in, not a delta-only folder. The same rule applies to Lectures: each `context.md`/notebook stands alone.

## Consequences
- Some duplication across consecutive classes' Project folders (accepted trade-off).
- Any class can be taught, reviewed, or graded in isolation.
- When a later class evolves an earlier class's project, the README should still narrate the improvement (before/after), even though the code is a full copy rather than a linked diff.
