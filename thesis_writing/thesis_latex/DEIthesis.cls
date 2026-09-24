%% DEIthesis.cls
%% Cleaned-up LaTeX class for a Master's Thesis at University of Padua, DEI

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{DEIthesis}[2025 DEI Thesis Class]

%% Base class
\LoadClass[12pt, a4paper, oneside]{book}

%% Page Geometry
%% DEI margins: top 2 cm, bottom 2 cm, inner 3 cm, outer 2 cm.
%% inner/outer (not left/right) so the layout stays correct if twoside is
%% ever enabled. Original template values were top=2.5, bottom=3, left/right=2.5.
\RequirePackage[
  top=2cm,
  bottom=2cm,
  inner=3cm,
  outer=2cm,
  headheight=14pt,
  footskip=1.5cm,
  a4paper
]{geometry}

%% Fonts and Encoding
%% DEI compliance: Times New Roman, 12 pt. The 12 pt size comes from the
%% \LoadClass option above. For the typeface, newtx provides Times-compatible
%% text (TeX Gyre Termes / Nimbus Roman No9 L) plus a matching math font, and
%% is metrically equivalent to Times New Roman under pdfLaTeX. The original
%% template used newpxtext/newpxmath, which is a Palatino clone -- not Times.
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{amsmath}
%% amssymb (for \checkmark, used in Chapter 3's method-variant table) must be
%% loaded BEFORE newtxmath: newtxmath redefines several AMS math symbol
%% fonts, and loading amssymb afterwards re-declares \Bbbk and fails with
%% "Command \Bbbk already defined" (found during the 2026-09-06 Chapter 2/3
%% LaTeX integration pass -- newtxmath's own documentation notes this
%% ordering requirement).
\RequirePackage{amssymb}
\RequirePackage{newtxtext}
\RequirePackage{newtxmath}

%% Essential Packages
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{setspace}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{watermark}
\RequirePackage{titlesec}
\RequirePackage{xcolor}
\RequirePackage{adjustbox}
\RequirePackage{booktabs}
\RequirePackage{csquotes}
\RequirePackage{colorprofiles}
%% PDF/A: pdfx loads hyperref itself and sets it up for PDF/A output, so
%% hyperref is NOT \RequirePackage'd separately (avoids an option clash).
%% Document metadata is read from main.xmpdata (\jobname.xmpdata).
\RequirePackage[a-2b]{pdfx}
\hypersetup{hidelinks}
%% Citation style (2026-09-07 literature-extension pass): switched from
%% style=authoryear to style=numeric-comp. No DEI/UniPD template requirement
%% mandates author-year (see DEI_COMPLIANCE.md item 8 -- authoryear was only
%% a prior choice to match how Chapter 2 was originally drafted); numeric is
%% the standard choice for a thesis that otherwise cites via \cite/\parencite
%% rather than \textcite-driven author-year prose. "-comp" compresses
%% consecutive numeric ranges in multi-key citations (e.g. [7,8,9] -> [7-9]).
%% maxcitenames/uniquename/uniquelist no longer affect in-text citations
%% under a numeric style but are kept: maxbibnames/giveninits still shape
%% how author names are printed in the bibliography list itself.
\RequirePackage[
  backend=biber,
  style=numeric-comp,
  maxcitenames=2,
  maxbibnames=99,
  uniquename=false,
  uniquelist=false,
  giveninits=true,
  sorting=none,
]{biblatex}
\addbibresource{references.bib}
\RequirePackage{listings}
\RequirePackage{acronym}

%% Added for Chapter 2/3 LaTeX integration pass (2026-09-06): original
%% TikZ conceptual figures in Chapter 2 (continual-learning taxonomy, LoRA
%% diagram) and \checkmark in Chapter 3's method-variant table. No other
%% chapter requires these. tabularx gives the related-work synthesis table
%% in Chapter 2 a controlled total width instead of relying on manual
%% column widths.
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc}
\RequirePackage{tabularx}

%% DEI compliance: 1.5 line spacing for the main body.
%% setspace automatically keeps footnotes, floats and captions single-spaced;
%% the title page sets its own spacing locally in \maketitle below.
\onehalfspacing



%% Define UniPD Red
\definecolor{UniPDColor}{RGB}{155, 0, 20}

%% Chapter, Section, Subsection Formatting
%% Major titles (chapter numbers/titles, and every \chapter*-based front-
%% matter/backmatter page: Abstract, Declaration of Originality,
%% Acknowledgements, Preface, Contents, List of Figures, List of Tables,
%% Bibliography) are left-aligned (\raggedright) rather than the template's
%% original \filleft (visually right-oriented). Style/class-level change so
%% every such page inherits it automatically instead of per-page hacks.
\titleformat{\chapter}[display]
  {\Huge\raggedright}
  {\minsizebox{!}{70pt}{\colorbox{UniPDColor}{\color{white} \thechapter}}}
  {1.2ex}
  {}

\titleformat{\section}[hang]
  {\Large}
  {\colorbox{UniPDColor}{\color{white} \thesection}}
  {1ex}
  {\scshape}

\titleformat{\subsection}[hang]
  {\large}
  {\colorbox{UniPDColor}{\color{white} \thesubsection}}
  {1ex}
  {\scshape}

\titleformat{\subsubsection}[hang]
  {\normalsize}
  {}
  {0ex}
  {\bfseries\scshape}

%% Fancyhdr Setup
%% Running header left-aligned (document is oneside, so the template's
%% original mirrored RO/LE pair rendered the chapter mark at the right
%% margin on some pages -- e.g. "CHAPTER 2. LITERATURE REVIEW" flush right).
%% A single \fancyhead[L] keeps it consistently at the left margin on every
%% page, as part of the page style rather than a per-page hack.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\leftmark}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}

%% Chapter-opening pages use the `plain' style, which \maketitle empties for
%% the front matter. From \mainmatter on, `plain' shows the page number at
%% the bottom centre (no running header); explicit \thispagestyle{empty}
%% pages (\blankpage) are unaffected.
\g@addto@macro\mainmatter{%
  \fancypagestyle{plain}{%
    \fancyhf{}%
    \fancyfoot[C]{\thepage}%
    \renewcommand{\headrulewidth}{0pt}%
    \renewcommand{\footrulewidth}{0pt}%
  }%
}

%% Title Page
\def\title#1{\gdef\@title{#1}}
\def\author#1{\gdef\@author{#1}}
\def\studentId#1{\gdef\@studentId{#1}}
\def\advisor#1{\gdef\@advisor{#1}}
\def\university#1{\gdef\@university{#1}}
\def\department#1{\gdef\@department{#1}}
\def\mastername#1{\gdef\@mastername{#1}}
\def\academicYear#1{\gdef\@academicYear{#1}}

%% Defaults for the DEI frontespizio (override in the master file if needed)
\gdef\@university{University of Padova}
\gdef\@department{Department of Information Engineering}

\renewcommand{\maketitle} {
  \newgeometry{
    top=1cm,
    bottom=1.5cm,
    inner=2cm,
    outer=2cm,
    headheight=1cm,
    includehead,
    includefoot,
  }

  \setlength{\headwidth}{\textwidth}
  \thiswatermark{\centering\put(-380,-700){\includegraphics[width=1.4\linewidth]{res/unipdbg}}}
  \thispagestyle{empty}

  \fancypagestyle{plain}{
    \fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
  }

  \begin{center}
    \begin{minipage}{0.45\textwidth}
      \raggedright
      \includegraphics[height=3cm]{res/ltunipd}
    \end{minipage}%
    \hfill
    \begin{minipage}{0.45\textwidth}
      \raggedleft
      \includegraphics[height=3cm]{res/dei}
    \end{minipage}
  \end{center}

  \begin{center}
    \vspace*{1.2cm}
    {\normalsize\textsc{\@university}} \\[0.6ex]
    {\normalsize\textsc{\@department}} \\[1.6ex]
    {\normalsize\textsc{Master's Degree in \@mastername}} \\[3ex]
    % Title font reduced from \LARGE to a custom size (2026-09-21, title-page
    % layout fix): the full title did not fit on one line at \LARGE within
    % the existing margins; \fontsize{19}{23} is the smallest reduction
    % (checked by rendering) that lets it fit on a single line while
    % remaining visibly the largest, most dominant text on the page.
    {\begin{spacing}{1.6}\fontsize{19}{23}\selectfont\bfseries \@title\end{spacing}}
    % NOTE (build repair, 2026-09-07): \end{spacing} (setspace) calls \par
    % internally, which drops out of horizontal mode -- the \\[10ex] that used
    % to follow here was therefore a genuine "There's no line here to end."
    % LaTeX Error on a from-scratch compile (silently swallowed only because
    % nonstopmode auto-ignores it and TeX recovers). \vspace{10ex} reproduces
    % the same visual gap and is valid in vertical mode.
    \vspace{10ex}
  \end{center}

  % NOTE (build repair, 2026-09-07): the three "\begin{spacing}{1}...
  % \end{spacing} \\" cells below each triggered a genuine "There's no line
  % here to end." LaTeX Error on a from-scratch compile -- \end{spacing}
  % (setspace) calls \par internally, which is invalid immediately before a
  % tabular row's \\ (silently swallowed only because nonstopmode auto-
  % ignores it). \setstretch{1} sets the same single-spacing stretch factor
  % without the environment's \par side effect, so it is safe inside a
  % tabular cell.
  \begin{center}
    \begin{tabular}{p{0.45\textwidth} p{0.45\textwidth}}
      \raggedright {\setstretch{1}\large Master Candidate} \\
      {\setstretch{1}\textbf{\Large \@author}} \\[0.4cm] \raggedright \large Student ID: \@studentId &
      \raggedleft {\setstretch{1}\large Advisor} \\
      \raggedleft \textbf{\Large \@advisor}
    \end{tabular}
  \end{center}

  \vfill
  \begin{center}
    {\normalsize \@university} \\[4ex]
    {\large \@academicYear}
  \end{center}

  \null
  \restoregeometry
}

%% ---- BACKMATTER ----
%% Acknowledgements now live in the front matter (frontmatter/thanks.tex,
%% \input directly from main.tex) rather than here, so \backmatter no longer
%% prints them -- only the bibliography, then the switch to \appendix.
\renewcommand{\backmatter}{
  \cleardoublepage
  \phantomsection

  % add references
  \addcontentsline{toc}{chapter}{\bibname}
  \printbibliography

  \cleardoublepage
  \phantomsection

  % start appendices
  \appendix
}

%% A single blank page: no header, no footer, no printed page number. Used
%% between front-matter pages (title page, Declaration of Originality,
%% Abstract) per the requested front-matter layout.
\newcommand{\blankpage}{%
  \clearpage
  \thispagestyle{empty}
  \mbox{}
  \clearpage
}

% -- CODE SNIPPETS --

% Colors for code snippets highlights
\definecolor{commentsColor}{RGB}{36, 161, 156}
\definecolor{numColor}{RGB}{71, 96, 114}
\definecolor{stringColor}{RGB}{205, 63, 62}
\definecolor{kwColor}{RGB}{248, 6, 204}
\definecolor{bgCodeColor}{RGB}{255, 249, 249}

% Style of code snippets
\lstdefinestyle{codeSnippet} {
  backgroundcolor=\color{bgCodeColor},
  commentstyle=\color{commentsColor},
  keywordstyle=\color{kwColor},
  numberstyle=\tiny\color{numColor},
  stringstyle=\color{stringColor},
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,         
  breaklines=true,                 
  captionpos=b,                    
  keepspaces=true,                 
  numbers=left,                    
  numbersep=5pt,                  
  showspaces=false,                
  showstringspaces=false,
  showtabs=false,                  
  tabsize=2
}

\lstset{style=codeSnippet}


%% Prevent unwanted blank pages while keeping page numbers
\makeatletter
\renewcommand{\cleardoublepage}{%
  \clearpage
  \if@twoside
    \ifodd\c@page
    \else
      \thispagestyle{fancy}\null\clearpage
    \fi
  \fi
}
\makeatother