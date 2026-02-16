\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesisDocumentclass}

% Basisoptionen für die Klasse
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

% Lade die Basis-Klasse report
\LoadClass[a4paper,12pt,twoside,BCOR10mm,DIV13,headinclude,footinclude=false,final,abstract=true,openright]{report}
% Ermögliche explizite Datum anzugeben
\makeatletter
\newcommand*{\specificDate}[1]{\@specificDate#1\@nil}
\newcommand*{\@specificDate}{}
\def\@specificDate#1.#2.#3\@nil{\begingroup
  \day=#1\relax
  \month=#2\relax
  \year=#3\relax
  \today
\endgroup}
% Allgemeine Pakete
\RequirePackage{ifthen}
\RequirePackage[a4paper,left=3cm,right=2.5cm,top=3cm,bottom=3cm]{geometry}
\RequirePackage{lipsum}
\RequirePackage[utf8]{inputenc}
\RequirePackage{datetime}
\RequirePackage[T1]{fontenc}
\RequirePackage{amsmath,amsthm,amsfonts}
\RequirePackage{graphicx}
\RequirePackage{hyperref}
\RequirePackage{autobreak}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{newpxtext}
\RequirePackage{newpxmath}
\RequirePackage{bm}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% Hyperref-Setup
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    filecolor=black,      
    urlcolor=black,
    citecolor=black,
    pdftitle={Latex Template},
    pdfpagemode=FullScreen,
}
% cleverref
\RequirePackage[nameinlink]{cleveref}
% aliascnt to make cleveref distinguish environments that share a counter
\RequirePackage{aliascnt}


% Schriftgröße und Kapitelstil
\def\largetitlestyle{\normalfont}
\titleformat{\chapter}[display]
    {\flushright}
    {\fontsize{80}{80}\selectfont\largetitlestyle\thechapter}
    {0pt}
    {\Huge}
\titlespacing*{\chapter}{0pt}{0pt}{2\baselineskip}

% Abschnittsüberschriften
\titleformat{\section}
    {\Large\bfseries}
    {\thesection.}
    {20pt}
    {}

\titleformat{\subsection}
    {\large\bfseries}
    {\thesubsection.}
    {5pt}
    {}

\titleformat{\subsubsection}
    {\bfseries}
    {}
    {0pt}
    {}
\titlespacing*{\subsubsection}{0pt}{\bigskipamount}{0pt}

% Seitenstil
\fancyhf{}
\pagestyle{fancy}
\fancyhead[OR]{\thepage}% ungerade Seiten, rechts
\fancyhead[OL]{\leftmark}% ungerade Seiten, links
\fancyhead[ER]{\leftmark}% gerade Seiten, rechts
\fancyhead[EL]{\thepage}% gerade Seiten, links
\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[R]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

% Leerzeichen und Abstände
\onehalfspacing

% Neue Umgebungen
\theoremstyle{definition}
\newtheorem{definition}{Definition}[chapter]

\newaliascnt{lemma}{definition}
\newtheorem{lemma}[lemma]{Lemma}
\aliascntresetthe{lemma}

\newaliascnt{proposition}{definition}
\newtheorem{proposition}[proposition]{Proposition}
\aliascntresetthe{proposition}

\newaliascnt{corollary}{definition}
\newtheorem{corollary}[corollary]{Corollary}
\aliascntresetthe{corollary}

\newaliascnt{theorem}{definition}
\newtheorem{theorem}[theorem]{Theorem}
\aliascntresetthe{theorem}

\newaliascnt{example}{definition}
\newtheorem{example}[example]{Example}
\aliascntresetthe{example}

\newaliascnt{remark}{definition}
\newtheorem{remark}[remark]{Remark}
\aliascntresetthe{remark}

\newcounter{assumptionCounter}
\newaliascnt{assumption}{assumptionCounter}
\newtheorem{assumption}[assumption]{Assumption}
\aliascntresetthe{assumption}

\crefname{theorem}{theorem}{theorems}
\Crefname{theorem}{Theorem}{Theorems}
\crefname{lemma}{lemma}{lemmas}
\Crefname{lemma}{Lemma}{Lemmas}
\crefname{proposition}{proposition}{propositions}
\Crefname{proposition}{Proposition}{Propositions}
\crefname{corollary}{corollary}{corollaries}
\Crefname{corollary}{Corollary}{Corollaries}
\crefname{example}{example}{examples}
\Crefname{example}{Example}{Examples}
\crefname{definition}{definition}{definitions}
\Crefname{definition}{Definition}{Definitions}
\crefname{algorithm}{algorithm}{algorithms}
\Crefname{algorithm}{Algorithm}{Algorithms}
\crefname{notation}{notation}{notations}
\Crefname{notation}{Notation}{Notations}
\crefname{assumption}{assumption}{assumptions}
\Crefname{assumption}{Assumption}{Assumptions}
\crefname{remark}{remark}{remarks}
\Crefname{remark}{Remark}{Remarks}

\renewcommand{\proofname}{Proof}

% Weitere Befehle
\newcommand{\sign}{\text{sign}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\E}{\mathbb{E}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Prob}{\mathbb{P}}

\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator{\proj}{\Pi_\mathcal{X}}
\DeclareMathOperator{\dist}{dist}
\DeclareMathOperator{\prox}{prox}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\dom}{dom}
\DeclareMathOperator{\graph}{graph}
\DeclareMathOperator{\interior}{int}
\DeclareMathOperator{\range}{range}

% Anpassung von \cleardoublepage
\let\origdoublepage\cleardoublepage
\renewcommand{\cleardoublepage}{%
  \clearpage
  \ifodd\value{page}%
  \else
    \thispagestyle{empty}%
    \hbox{}%
    \newpage
  \fi
}

\endinput