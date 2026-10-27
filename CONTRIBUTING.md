# Contributing

1. **Branch** from `main`: `feature/<name>-<topic>` (fixes: `fix/<name>-<topic>`, interface changes: `contracts/<topic>`).
2. **Stay in your files.** Each path has one owner (see `.github/CODEOWNERS`). Need something in someone else's
   module? Ask them, or open an issue.
3. **Test locally:** `ruff check .` and `pytest`.
4. **Open a PR** using the template: name the roadmap task, its definition of done, and paste test output.
5. **Review:** the track that uses your code reviews within 24 h. CI must be green.
6. **Merge:** the author squash-merges after approval, then deletes the branch.
7. **Interfaces:** `contracts.py` is frozen as v1.0 on Nov 1. After that, changes need a separate
   `contracts/` PR approved by all four members, with updated fixtures in the same PR.
8. **Integration test failures:** Mihir owns `tests/integration/`; the owner of the failing module fixes it in
   their own PR.
