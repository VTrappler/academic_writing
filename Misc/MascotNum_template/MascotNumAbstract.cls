% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % Template file for MascotNum 2016 Phd Student papers   % % %
% % % November 28th, 2014                                   % % %
% % % (c) Stefano Marelli, Bruno Sudret, Nicolas Durrande   % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MascotNumAbstract}[2014/10/27 MascotNumAbstract]

%% Article options
\DeclareOption{10pt}{
	\PassOptionsToClass{\CurrentOption}{article}
}

%% Fallback
\DeclareOption*{
	\ClassWarning{myclass}{Unknown option '\CurrentOption'}
}


%% Custom package options
\DeclareOption{sansserif}{
	\PassOptionsToPackage{\CurrentOption}{paxcommands}
}
\DeclareOption{neverindent}{
	\PassOptionsToPackage{\CurrentOption}{paxcommands}
}


%% Execute default options
\ExecuteOptions{10pt}

%% Process given options
\ProcessOptions\relax

%% Load base class (paper in a4 format)
\LoadClass[a4paper,twoside]{article}

% Page layout
\RequirePackage[left=1.2in,right=1.2in,top=1.2in,bottom=1.2in,headheight=13pt]{geometry}
\RequirePackage{filecontents}
\RequirePackage{graphicx}

% hyperlinks
%\RequirePackage[colorlinks=true, urlcolor=black]{hyperref}
%% And now come the headers

\RequirePackage{fancyhdr}
\fancypagestyle{firstpage}{
	%
	\fancyhf{}
	% clear all six fields
	%\renewcommand{\headheight}{13pt}
	\renewcommand{\headrulewidth}{0.5pt}
	\renewcommand{\footrulewidth}{0pt}
	\fancyhead[R]{\textbf{MascotNum Annual Conference, May 04-07
            2020, France}}
}
\fancypagestyle{followingpage}{
	%
	\fancyhf{}
	% clear all six fields
	%\fancyhead[RE,LO]{\show@committee, \show@date}
	%\renewcommand{\headheight}{13pt}
	\fancyhead[LE,RO]{\textbf{\textbf{MascotNum Annual Conference, May 04-07
            2020, France}}}
	%\fancyfoot[LE,LO]{\small \textbf{\runauthor{},  \runtitle{}}}
	%\fancyfoot[RE,RO]{\small - \thepage{} -}
	\renewcommand{\headrulewidth}{0.5pt}
	\renewcommand{\footrulewidth}{0pt}
}
\pagestyle{followingpage}
\setlength{\parindent}{0em}
\setlength{\parskip}{1.3ex plus 0.5ex minus 0.2ex}


%% useful header macros
%\newcommand{\shorttitle}[1]{\newcommand{\runtitle}{#1}}
%\newcommand{\shortauthor}[1]{\newcommand{\runauthor}{#1}}
\newcommand{\firstauthoraddress}[1]{\newcommand{\authaddr}{#1}}
%\newcommand{\masteruniversity}[1]{\newcommand{\masteruni}{#1}}
\newcommand{\phdtime}[1]{\newcommand{\phdt}{#1}}
\newcommand{\phduniversity}[1]{\newcommand{\univ}[1]{#1}}
%\newcommand{\correspondingauthor}[1]{\newcommand{\cauthor}{#1}}
\newcommand{\contactmail}[1]{\newcommand{\cmail}{#1}}
%\newcommand{\webaddress}[1]{\newcommand{\waddress}{#1}}
\newcommand{\supervisors}[1]{\newcommand{\superv}{#1}}
% setting page style on the first page (no fancy headings)
\newcommand{\biography}[1]{\newcommand{\biog}{#1}}

\AtBeginDocument{\thispagestyle{firstpage}}

%\renewcommand{\refname}{\normalfont\selectfont\normalsize References} 
\renewcommand{\refname}{\normalsize References} 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtBeginDocument{
\begin{minipage}{.1\textwidth}
~
\end{minipage}
\begin{minipage}{.89\textwidth}
	\vspace{.5cm}
%	\textsc{Ph.D. student presentation}\vspace{.3cm}\\
	\textbf{\Large{\@title}}\vspace{.5cm}\\
	\textsc{\@author}\\
	\textit{\univ}\vspace{.5cm}\\
\end{minipage}


\textbf{Supervisor(s):} \superv{}

\textbf{Ph.D. expected duration:} \phdt{}

\textbf{Adress}: \authaddr{}

\textbf{Email}: \cmail{}
%\textbf{URL}: \href{\waddress}{\nolinkurl{\waddress}}

%\textbf{Master:} \masteruni{}

%\textbf{Ph.D.} \phdt{}: \univ{}

\vspace*{1cm}
\textbf{Abstract: }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtEndDocument{
\vfill 

{\bf Short biography --} \biog{}

\vspace*{1cm}
}
%% Close the package
\endinput
