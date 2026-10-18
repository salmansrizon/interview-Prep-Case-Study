# 0003. Week 9 Class 1 deviates from Course Module.pdf's stated topic

## Status
Accepted

## Context
`Course Module.pdf` scopes Class 17 as "Image Processing basics": OpenCV kernels (blur/sharpen), grayscale, thresholding, contour detection, with the achievement being a Document Scanner (perspective-corrected photo). The repo's `week 9/Class 1 Project` instead already contains a fully built, production-quality app — "Emoji Cam" — which teaches Haar-cascade face detection, threaded camera capture, and alpha-blended overlay compositing (conceptually closer to Class 18's territory). Rebuilding a curriculum-accurate Document Scanner would discard working, polished material; keeping Emoji Cam without comment would leave a silent, confusing gap between the course document and the repo.

## Decision
Keep Emoji Cam as Week 9 Class 1's Project. Rewrite `week 9/Class 1 Lecture/context.md` to teach what Emoji Cam actually demonstrates — real-time face detection (Haar Cascade), threaded frame capture, and image compositing — rather than the PDF's kernels/threshold/contour/Document-Scanner topic.

## Consequences
- Week 9 Class 1's Lecture and Project no longer match `Course Module.pdf` verbatim; this ADR is the record of why.
- Contour detection, thresholding, and perspective warp (the PDF's original Class 17 content) are not covered anywhere in the bootcamp unless reintroduced later — acceptable since nothing downstream currently depends on them.
- Any future audit comparing the repo against the course PDF should be pointed at this ADR rather than treated as an unexplained bug.
