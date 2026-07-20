# Release process

1. Confirm the tree contains no restricted data, credentials, private paths, real item-level
   predictions, model weights, or unlicensed third-party files.
2. Run the full Python, structured-file, link, and synthetic verification suite in a clean
   environment.
3. Review the claim-to-evidence boundary and changelog.
4. Tag an immutable semantic version from `main` only after CI is green.
5. Publish release notes that label the software maturity and evidence level accurately.
6. For a stable archival release, create a DOI-backed archive and update `CITATION.cff`.

Alpha tags may change APIs. A stable release requires at least one externally reproduced install,
resolved roadmap blockers, and maintainer review of scientific-default behavior.
