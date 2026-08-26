# Verseprint accessibility and interaction audit

**Audit date:** 25 August 2026  
**Build:** local production build at `http://127.0.0.1:8790/`  
**Standard used:** WCAG 2.1 AA-oriented manual and programmatic checks  
**Verdict:** PASS for the checks below

| Check | Evidence | Result |
| --- | --- | --- |
| Page structure | One `main` landmark per view; labelled primary `nav`; one view-level `h1`; ordered subordinate headings | PASS |
| Keyboard navigation | Repertoire SVG edges and nodes and reference nodes accept Enter/Space; selected evidence changes and exposes `aria-pressed` | PASS |
| Programmatic names | Graph nodes expose concise action/evidence `aria-label` values; the home control has `aria-label="Verseprint home"`; graph containers have contextual labels | PASS |
| Focus visibility | Global `:focus-visible` styling and explicit SVG focus strokes are present | PASS |
| Target size | Every visible HTML/SVG action measured at least 44 CSS px in both dimensions after adding a 44 px transparent hit target to edge explanations | PASS |
| Muted text contrast | `#626a75` on `#f1eee6`: 4.72:1 | PASS |
| Blue text contrast | `#1f5fdc` on `#f1eee6`: 4.86:1 | PASS |
| White on blue | `#ffffff` on `#1f5fdc`: 5.63:1 | PASS |
| Dark on accent | `#11141a` on `#e5ff3d`: 16.42:1 | PASS |
| Mobile horizontal overflow | Home, reference, and rhyme views: `scrollWidth == clientWidth == 375` at the test viewport | PASS |
| Meaningful reference path | `GALI → 上海 → 法老` displays song-unit support, shrunken enrichment, uncertainty/FDR, and a non-biographical boundary | PASS |
| Meaningful rhyme result | Input `你` yields family `I`, five observed transition options, descriptive repertoire evidence, held-out performance, switch weakness, and no-personalization notice | PASS |

The interface publishes no lyric lines, source identifiers, private membership rows, or reviewer context. Text alternatives describe graph meaning; they do not restate raw geometry. This audit does not replace testing with assistive-technology users and does not certify the surrounding browser or operating system.
