# SelectivePrep

Browser-based NSW Selective practice test environment.

## 2021 test

Open `2021/index.html` in a browser.

It includes:
- Reading, Mathematical Reasoning and Thinking Skills
- independent timer for each subject
- previous/next navigation
- answered/unanswered state
- flag for review
- local progress persistence
- submit now or review results later
- correct/incorrect/unanswered breakdown
- review of incorrect questions
- per-subject reset

Large question and passage images are stored in chunked JavaScript asset files under `2021/assets/`, so future UI and behaviour changes can update the existing codebase without rebuilding the test content from scratch.

## Source

The reusable page template is stored at `src/index.template.html`.
