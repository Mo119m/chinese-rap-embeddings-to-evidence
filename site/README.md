# Results application source

The repository root `index.html` is the self-contained, no-server version intended for readers and colleagues.

This directory contains the richer React/Vite source used for development and local research inspection.

```bash
pnpm install
pnpm dev
```

Then open the local URL printed by the development server.

The application bundles only the copyright-safe aggregate files in `app/data/`; `public/data/researchData.json` is the matching public export used for validation and reuse. It does not include lyric text, full written lines, song/chunk identifiers, embeddings, row-level lyric-content hashes, or reviewer contexts. Its manifest contains only a file-level integrity checksum.
