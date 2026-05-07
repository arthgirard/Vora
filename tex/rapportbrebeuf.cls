% Classe de rapport – Collège Jean-de-Brébeuf

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rapportbrebeuf}[2026/05/06 Classe de rapport Brébeuf]

% Classe de base
\LoadClass[11pt, letterpaper]{article}

% ENCODAGE ET LANGUE
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[french]{babel}

% MISE EN PAGE
\RequirePackage[
  top=2.54cm, bottom=2.54cm,
  left=3.17cm, right=3.17cm
]{geometry}
\RequirePackage{fancyhdr}

% TYPOGRAPHIE
\RequirePackage{microtype}
\RequirePackage{lmodern}
\RequirePackage{setspace}
\onehalfspacing

% IMAGES
\RequirePackage{graphicx}
\RequirePackage{float}

% MATHS
\RequirePackage{amsmath, amssymb}

% SECTIONS – FORMATAGE SANS NUMÉROTATION
\RequirePackage{titlesec}

% Sections principales
\titleformat{\section}
  [block]{\normalfont\Large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{24pt}{12pt}

% Sous-sections
\titleformat{\subsection}
  [hang]{\normalfont\large\bfseries}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{18pt}{6pt}

% Sous-sous-sections
\titleformat{\subsubsection}
  [hang]{\normalfont\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\subsubsection}{0pt}{12pt}{4pt}

% Suppression de la numérotation automatique
\setcounter{secnumdepth}{0}

% LIENS
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{xurl}
\urlstyle{same}

% DIVERS
\RequirePackage{hanging}
\RequirePackage{parskip}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}

% BIBLIOGRAPHIE
\RequirePackage{csquotes}
\RequirePackage[
    backend=biber,
    style=ieee, % Active le format IEEE
    urldate=iso, % Format de date ISO (AAAA-MM-DD)
    date=iso, % Format de date ISO
    seconds=true % Précision des dates si disponible
]{biblatex}

% VARIABLES DE LA PAGE TITRE
% Chaque champ est défini vide par défaut
\newcommand{\@auteurs}{}
\newcommand{\@titre}{}
\newcommand{\@soustitre}{}
\newcommand{\@presenteA}{}
\newcommand{\@cours}{}
\newcommand{\@groupe}{}
\newcommand{\@institution}{}
\newcommand{\@departement}{}
\newcommand{\@dateremise}{}

% Commandes de saisie des champs
\newcommand{\auteurs}[1]{\renewcommand{\@auteurs}{#1}}
\newcommand{\titre}[1]{\renewcommand{\@titre}{#1}}
\newcommand{\soustitre}[1]{\renewcommand{\@soustitre}{#1}}
\newcommand{\presenteA}[1]{\renewcommand{\@presenteA}{#1}}
\newcommand{\cours}[1]{\renewcommand{\@cours}{#1}}
\newcommand{\groupe}[1]{\renewcommand{\@groupe}{#1}}
\newcommand{\institution}[1]{\renewcommand{\@institution}{#1}}
\newcommand{\departement}[1]{\renewcommand{\@departement}{#1}}
\newcommand{\dateremise}[1]{\renewcommand{\@dateremise}{#1}}

% COMMANDE DE PAGE TITRE
% Génère la page titre à partir des variables définies ci-dessus
\newcommand{\fairePageTitre}{%
  \thispagestyle{empty}
  \begin{center}
    \@auteurs

    \vfill

    \@titre\\[0.2cm]
    \@soustitre

    \vfill

    Présenté à\\[0.2cm]
    \@presenteA\\[0.2cm]
    \@cours\\[0.2cm]
    \@groupe

    \vfill

    \@institution\\[0.2cm]
    \@departement\\[0.2cm]
    \@dateremise
  \end{center}
}

% COMMANDE DE TABLE DES MATIÈRES
% Numérotation romaine pour les pages préliminaires
\newcommand{\faireTDM}{%
  \newpage
  \tableofcontents
  \pagenumbering{roman}
}

% COMMANDE DE DÉBUT DU CORPS
% Change les marges et bascule vers la numérotation arabe
% Les numéros de page alternent à gauche (pair) et à droite (impair)

% Style de page sans en-tête – pagination alternée en pied de page uniquement
\fancypagestyle{corpsprincipal}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \fancyfoot[R]{\ifodd\value{page}\thepage\fi}
  \fancyfoot[L]{\ifodd\value{page}\else\thepage\fi}
}

\newcommand{\debutCorps}{%
  \newpage
  \newgeometry{top=3cm, bottom=2.5cm, left=4cm, right=2.5cm}
  \pagenumbering{arabic}
  \setcounter{page}{1}
  \pagestyle{corpsprincipal}
}
