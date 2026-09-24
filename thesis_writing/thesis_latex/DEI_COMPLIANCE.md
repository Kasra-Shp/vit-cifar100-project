# DEI Compliance Audit — `thesis_writing/thesis_latex/`

**Target:** University of Padua · Department of Information Engineering (DEI) ·
Master's Degree in Control Systems Engineering.

**Base:** the actual DEI template archive
`thesis_writing/thesis-template-dei-unipd.zip`, extracted in place.
Master file `main.tex`; document class `DEIthesis.cls` (custom class shipped
*inside* the template — `\LoadClass[12pt,a4paper,oneside]{book}`).

**Toolchain status (2026-08-31):** no LaTeX toolchain installed on this
machine (`pdflatex`, `xelatex`, `lualatex`, `biber`, `bibtex`, `latexmk`,
`tectonic` all absent). Nothing below has been compiled. Items are marked
**STATICALLY VERIFIED** (checked by reading the sources) or **REQUIRES REAL
LATEX COMPILATION**.

---

## Compliance table

| Requirement | Target value | Original template value | Final value | File modified | Status |
|---|---|---|---|---|---|
| Paper size | A4 | `a4paper` (class + geometry) | `a4paper` (unchanged) | — | STATICALLY VERIFIED |
| Main text size | 12 pt | `12pt` (`\LoadClass`) | `12pt` (unchanged) | — | STATICALLY VERIFIED |
| Text font | Times New Roman | `newpxtext` (Palatino clone) | `newtxtext` (Times-compatible: TeX Gyre Termes / Nimbus Roman) | `DEIthesis.cls` | STATICALLY VERIFIED (config); font *rendering* REQUIRES REAL LATEX COMPILATION |
| Math font | Times-compatible | `newpxmath` (Palatino math) | `newtxmath` (Times-compatible math) | `DEIthesis.cls` | STATICALLY VERIFIED (config); REQUIRES REAL LATEX COMPILATION |
| Line spacing (body) | 1.5 | `setspace` loaded, **no** `\onehalfspacing` | `\onehalfspacing` added | `DEIthesis.cls` | STATICALLY VERIFIED |
| Top margin | 2 cm | `top=2.5cm` | `top=2cm` | `DEIthesis.cls` | STATICALLY VERIFIED (value); effective layout REQUIRES REAL LATEX COMPILATION |
| Bottom margin | 2 cm | `bottom=3cm` | `bottom=2cm` | `DEIthesis.cls` | STATICALLY VERIFIED |
| Inner margin | 3 cm | `left=2.5cm` | `inner=3cm` | `DEIthesis.cls` | STATICALLY VERIFIED |
| Outer margin | 2 cm | `right=2.5cm` | `outer=2cm` | `DEIthesis.cls` | STATICALLY VERIFIED |
| Language | English | `\usepackage[english]{babel}` | unchanged | — | STATICALLY VERIFIED |
| University wording | University of Padua | `\university{University of Padua}` (in `main.tex`) | `\university{University of Padua}` + class default; printed on frontespizio | `DEIthesis.cls`, `main.tex` | STATICALLY VERIFIED |
| Department wording | Department of Information Engineering | *not present as text* (only the DEI logo) | `\department{Department of Information Engineering}` field + printed on frontespizio | `DEIthesis.cls`, `main.tex` | STATICALLY VERIFIED |
| Degree programme wording | Master's Degree in Control Systems Engineering | `\mastername{Computer Engineering}`, printed as "Master's Thesis in …" | `\mastername{Control Systems Engineering}`, printed as "Master's Degree in Control Systems Engineering" | `DEIthesis.cls`, `main.tex` | STATICALLY VERIFIED |
| Frontespizio | Institutional English title page | template title page (logos, degree, title, candidate/advisor, university, AY) | same layout + University/Department/Degree text lines added; placeholders for name/ID/advisor/title | `DEIthesis.cls`, `main.tex` | STATICALLY VERIFIED (structure); PENDING personal data + REQUIRES REAL LATEX COMPILATION |
| Bibliography system | biblatex | `biblatex` | `biblatex` (unchanged) | — | STATICALLY VERIFIED |
| Bibliography backend | biber | `backend=biber` | `backend=biber` + `style=authoryear` (+ `maxcitenames=2, maxbibnames=99, giveninits`) added to match Chapter 2 | `DEIthesis.cls` | STATICALLY VERIFIED (config); output REQUIRES REAL LATEX COMPILATION |
| PDF/A system | present, single mechanism | `pdfx` (no version option) + `colorprofiles` | `\usepackage[a-2b]{pdfx}` + `colorprofiles` + `main.xmpdata` metadata | `DEIthesis.cls`, `main.xmpdata` (new) | PDF/A CONFIGURATION PRESENT — COMPILE/VALIDATION PENDING |
| oneside / twoside | preserve template intent | `oneside` | `oneside` (unchanged) | — | STATICALLY VERIFIED |

---

## Section-by-section notes

### A. Page format
`\LoadClass[12pt, a4paper, oneside]{book}` and `geometry` both request A4;
`12pt` is the base size. No change needed.

### B. Margins
Defined once, in `DEIthesis.cls`, in the `\RequirePackage[...]{geometry}`
call near the top.
Original: `top=2.5cm, bottom=3cm, left=2.5cm, right=2.5cm`.
Final: `top=2cm, bottom=2cm, inner=3cm, outer=2cm` (`headheight=14pt`,
`footskip=1.5cm` kept). `inner/outer` chosen (not `left/right`) so the
binding margin stays correct if `twoside` is ever enabled.
The title page uses its own local `\newgeometry{... inner=2cm, outer=2cm ...}`
inside `\maketitle`; left as designed by the template.

### C. Line spacing
`setspace` was already loaded but never activated. Added `\onehalfspacing`
in the class after the package block. `setspace` automatically keeps
footnotes, floats and captions single-spaced; the title page sets spacing
locally. Bibliography spacing follows `biblatex` defaults.

### D. Font
- **Selected compiler target:** pdfLaTeX (the template is built on
  `inputenc`/`fontenc` + `newpx`; `pdfx` PDF/A works most reliably here).
- **Text font:** `newtxtext` → TeX Gyre Termes / Nimbus Roman No9 L, a
  metric-compatible equivalent of Times New Roman for pdfLaTeX.
- **Math font:** `newtxmath` (matching Times-style math).
- **Is it literally "Times New Roman"?** No — it is the standard
  **Times-compatible fallback** for pdfLaTeX. True *Times New Roman* would
  require XeLaTeX/LuaLaTeX + `fontspec` (`\setmainfont{Times New Roman}`),
  which would mean replacing the template's `inputenc`/`fontenc`/`newpx`
  stack and re-checking the `pdfx` pipeline. No external font files were
  downloaded, embedded or redistributed.
- Only one font system is loaded (`newtx`); `newpx` was removed.

### E. English configuration
`babel` is `english`. Contents / List of Figures / List of Tables /
Bibliography / chapter headings / captions all resolve to English strings
via `babel`. Frontespizio text lines are English. No Italian placeholder
labels remain in the class or master file.

### F. Frontespizio
Layout preserved (UniPD + DEI logos, background watermark, degree line,
title, "Master Candidate" / "Advisor" block with Student ID, university and
academic year). Added three centered text lines above the title:
`University of Padua` / `Department of Information Engineering` /
`Master's Degree in <mastername>`, plus a new `\department{}` field in the
class. Personal fields in `main.tex` are **clearly-marked placeholders**
(`PLACEHOLDER -- …`) for title, candidate name, student ID and supervisor —
none invented. Academic year set to `2025/2026` (adjust if needed).

### G. PDF/A
- **Mechanism:** `pdfx` (loads `hyperref` itself and configures it for
  PDF/A) + `colorprofiles` (sRGB output intent). Single mechanism — no
  competing PDF/A package added.
- **Version:** original template passed **no** version option to `pdfx`;
  set to **`a-2b`** (PDF/A-2b). `a-1b` is the stricter alternative if the
  submission portal requires it.
- **Metadata:** new `main.xmpdata` provides `\Title`, `\Author`,
  `\Language{en-US}`, `\Subject`, `\Keywords`, `\Publisher` (placeholders to
  be filled with the frontespizio data).
- **Change from original:** removed the standalone `\RequirePackage{hyperref}`
  so `pdfx` owns the hyperref setup (avoids an option clash); added
  `\hypersetup{hidelinks}`.
- **Validation:** **NOT TESTED.** Status: *PDF/A CONFIGURATION PRESENT —
  COMPILE/VALIDATION PENDING.* Real PDF/A conformance must be checked with
  veraPDF or Acrobat Preflight after a successful build.

### H. oneside / twoside
Template ships `oneside`; **kept**. No DEI requirement here mandates
`twoside`. Under `oneside`, `geometry`'s `inner` maps to the left margin and
`outer` to the right, so the 3 cm / 2 cm split renders as left 3 cm /
right 2 cm on every page. If `twoside` is later enabled, `inner`/`outer`
already make the binding side correct on both recto and verso — no further
change needed.

### I. Headings / visual style
Unchanged. The template's `titlesec` chapter/section styling (UniPD-red
number boxes, `\filleft` display chapters) is preserved. Only
paper/font/margins/spacing/language/frontespizio/bibliography/PDF-A were
touched.

---

## Known template defects fixed (template was **not** compliant as shipped)

1. **Font was Palatino, not Times** — `newpxtext`/`newpxmath` → `newtxtext`/`newtxmath`.
2. **No line spacing applied** — `setspace` loaded but `\onehalfspacing` missing → added.
3. **Margins non-compliant** — 2.5/3/2.5/2.5 → 2/2/3(inner)/2(outer).
4. **`\backmatter` called undefined `\acknowledgments` / `\acknowledgmentsname`** —
   would break compilation → both `\providecommand`'d in the class
   (`\acknowledgments` now `\input`s `frontmatter/thanks.tex`, which is
   therefore no longer `\input` from the frontmatter).
5. **`pdfx` had no PDF/A version option** — added `a-2b`.
6. **Wrong degree name** — `Computer Engineering` → `Control Systems Engineering`.
7. **No Department / University text on the frontespizio** — added.
8. **biblatex had no style** — Chapter 2 uses `\parencite`/`\textcite`
   (author-year); added `style=authoryear` (+ name options).

---

## Still requires a real LaTeX compilation

- Actual font substitution and embedding (`newtx` present in the TeX install).
- `pdfx` + `biblatex` + `newtxmath` + `titlesec` package interaction.
- PDF/A-2b conformance (veraPDF / Preflight).
- Effective page geometry, ToC/LoF/LoT generation, all `\ref` numbers.
- `biber` run over `references.bib` (31 entries).
