latex_content = r"""
\documentclass[11pt, a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage{fancyhdr}

\definecolor{secblue}{RGB}{26, 71, 111}
\definecolor{notegreen}{RGB}{40, 120, 60}
\definecolor{explgray}{RGB}{100, 100, 100}

\newcommand{\note}[1]{{\small\color{notegreen}\textit{#1}}}
\newcommand{\expl}[1]{{\small\color{explgray}\textit{#1}}}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textit{GP Regression — Worked Example}}
\lhead{\textit{Arif's GP Study}}
\cfoot{\thepage}

\title{\color{secblue}\textbf{Gaussian Process Regression}\\[0.3em]
\Large Fully Worked Example with Actual Numbers}
\author{}
\date{}

\begin{document}
\maketitle
\thispagestyle{fancy}

% ============================================================
\section*{\color{secblue}Setup}

One soil sample. We measured reflectance at 2 wavelengths. We want to predict reflectance at 3 new wavelengths.

\begin{center}
\begin{tabular}{cc}
\toprule
\textbf{Wavelength (nm)} & \textbf{Reflectance} \\
\midrule
400 & 0.10 \\
600 & 0.30 \\
\bottomrule
\end{tabular}
\hspace{2cm}
\begin{tabular}{cc}
\toprule
\textbf{Wavelength (nm)} & \textbf{Reflectance} \\
\midrule
450 & ? \\
500 & ? \\
550 & ? \\
\bottomrule
\end{tabular}
\end{center}

\textbf{Kernel:} RBF with length scale $\ell = 100$ nm

\[
k(x, x') = \exp\!\left( -\frac{(x - x')^2}{2\,\ell^2} \right)
\]

\expl{``How similar are two wavelengths? Closer $\Rightarrow$ more similar $\Rightarrow$ higher value (max = 1).''}


% ============================================================
\section*{\color{secblue}Step 1: $\mathbf{K_{\text{train,train}}}$ — How Observed Points Relate to Each Other}

Compute the kernel between every pair of observed wavelengths. Result: a $2 \times 2$ matrix.

\begin{align*}
k(400, 400) &= \exp\!\left(-\frac{(0)^2}{2 \cdot 100^2}\right) = e^{0} = \mathbf{1.0000}
  & \expl{Same point $\rightarrow$ perfect correlation} \\[6pt]
k(400, 600) &= \exp\!\left(-\frac{(200)^2}{2 \cdot 100^2}\right) = e^{-2} = \mathbf{0.1353}
  & \expl{200\,nm apart $\rightarrow$ weak correlation} \\[6pt]
k(600, 600) &= e^{0} = \mathbf{1.0000}
  & \expl{Same point $\rightarrow$ perfect correlation}
\end{align*}

\[
K_{\text{train,train}} =
\begin{pmatrix} 1.0000 & 0.1353 \\ 0.1353 & 1.0000 \end{pmatrix}
\]

\note{400\,nm and 600\,nm are 200\,nm apart, so their correlation is only 0.14 — they don't know much about each other.}


% ============================================================
\section*{\color{secblue}Step 2: $\mathbf{K_{\text{pred,train}}}$ — How Prediction Points Relate to Observed Points}

This is the most important matrix. It tells us how much each prediction point ``listens to'' each observation.

\begin{align*}
k(450, 400) &= \exp\!\left(-\frac{50^2}{2 \cdot 100^2}\right) = e^{-0.125} = \mathbf{0.8825}
  & \expl{450 close to 400 $\rightarrow$ high similarity} \\[4pt]
k(450, 600) &= \exp\!\left(-\frac{150^2}{2 \cdot 100^2}\right) = e^{-1.125} = \mathbf{0.3247}
  & \expl{450 far from 600 $\rightarrow$ low similarity} \\[8pt]
k(500, 400) &= \exp\!\left(-\frac{100^2}{2 \cdot 100^2}\right) = e^{-0.5} = \mathbf{0.6065}
  & \expl{500 medium distance from 400} \\[4pt]
k(500, 600) &= e^{-0.5} = \mathbf{0.6065}
  & \expl{500 equally far from 600} \\[8pt]
k(550, 400) &= e^{-1.125} = \mathbf{0.3247}
  & \expl{550 far from 400 $\rightarrow$ low similarity} \\[4pt]
k(550, 600) &= e^{-0.125} = \mathbf{0.8825}
  & \expl{550 close to 600 $\rightarrow$ high similarity}
\end{align*}

\[
K_{\text{pred,train}} =
\begin{pmatrix}
0.8825 & 0.3247 \\
0.6065 & 0.6065 \\
0.3247 & 0.8825
\end{pmatrix}
\leftarrow
\begin{array}{l}
\text{450\,nm: mostly listens to 400\,nm} \\
\text{500\,nm: listens equally to both} \\
\text{550\,nm: mostly listens to 600\,nm}
\end{array}
\]


% ============================================================
\section*{\color{secblue}Step 3: $\mathbf{K_{\text{pred,pred}}}$ — How Prediction Points Relate to Each Other}

Needed for computing uncertainty. Same kernel, applied between prediction points.

\[
K_{\text{pred,pred}} =
\begin{pmatrix}
1.0000 & 0.8825 & 0.6065 \\
0.8825 & 1.0000 & 0.8825 \\
0.6065 & 0.8825 & 1.0000
\end{pmatrix}
\]

\note{This represents our prior uncertainty — before seeing any data.}


% ============================================================
\section*{\color{secblue}Step 4: Invert $K_{\text{train,train}}$}

Standard $2\times 2$ matrix inversion:

\[
\det = 1.0 \times 1.0 - 0.1353 \times 0.1353 = 0.9817
\]

\[
K_{\text{train,train}}^{-1} = \frac{1}{0.9817}
\begin{pmatrix} 1.0000 & -0.1353 \\ -0.1353 & 1.0000 \end{pmatrix}
=
\begin{pmatrix} 1.0186 & -0.1378 \\ -0.1378 & 1.0186 \end{pmatrix}
\]


% ============================================================
\section*{\color{secblue}Step 5: Posterior Mean — THE PREDICTION}

\[
\boldsymbol{\mu}_{\text{pred}} = K_{\text{pred,train}} \;\cdot\; K_{\text{train,train}}^{-1} \;\cdot\; \mathbf{y}_{\text{train}}
\]

\subsection*{Sub-step A: $\boldsymbol{\alpha} = K_{\text{train,train}}^{-1} \cdot \mathbf{y}_{\text{train}}$}

\[
\boldsymbol{\alpha} =
\begin{pmatrix} 1.0186 & -0.1378 \\ -0.1378 & 1.0186 \end{pmatrix}
\begin{pmatrix} 0.10 \\ 0.30 \end{pmatrix}
=
\begin{pmatrix}
1.0186 \times 0.10 + (-0.1378) \times 0.30 \\
(-0.1378) \times 0.10 + 1.0186 \times 0.30
\end{pmatrix}
=
\begin{pmatrix} 0.0605 \\ 0.2918 \end{pmatrix}
\]

\expl{$\alpha_1 = 0.0605$: effective weight from the 400\,nm observation}\\
\expl{$\alpha_2 = 0.2918$: effective weight from the 600\,nm observation}

\subsection*{Sub-step B: $\boldsymbol{\mu}_{\text{pred}} = K_{\text{pred,train}} \cdot \boldsymbol{\alpha}$}

\begin{align*}
\mu_{450} &= 0.8825 \times 0.0605 + 0.3247 \times 0.2918 = 0.0534 + 0.0948 = \mathbf{0.1482} \\
  & \expl{Close to 400\,nm\;(0.10) $\rightarrow$ pulled low} \\[6pt]
\mu_{500} &= 0.6065 \times 0.0605 + 0.6065 \times 0.2918 = 0.0367 + 0.1770 = \mathbf{0.2137} \\
  & \expl{Equidistant $\rightarrow$ right between the two observed values} \\[6pt]
\mu_{550} &= 0.3247 \times 0.0605 + 0.8825 \times 0.2918 = 0.0197 + 0.2575 = \mathbf{0.2772} \\
  & \expl{Close to 600\,nm\;(0.30) $\rightarrow$ pulled high}
\end{align*}


% ============================================================
\section*{\color{secblue}Step 6: Posterior Covariance — THE UNCERTAINTY}

\[
\Sigma_{\text{pred}} = K_{\text{pred,pred}} - K_{\text{pred,train}} \;\cdot\; K_{\text{train,train}}^{-1} \;\cdot\; K_{\text{pred,train}}^\top
\]

\expl{``Start with prior uncertainty, subtract what we learned from data.''}

\[
\Sigma_{\text{pred}} =
\begin{pmatrix}
0.1783 & 0.2374 & 0.1445 \\
0.2376 & 0.3520 & 0.2376 \\
0.1445 & 0.2374 & 0.1783
\end{pmatrix}
\]

Standard deviations (square root of diagonal):

\begin{center}
\begin{tabular}{ccc}
\toprule
\textbf{Wavelength} & \textbf{Variance} & $\boldsymbol{\sigma}$ \\
\midrule
450\,nm & 0.1783 & 0.4222 \\
500\,nm & 0.3520 & 0.5933 \\
550\,nm & 0.1783 & 0.4222 \\
\bottomrule
\end{tabular}
\end{center}

\note{500\,nm has the MOST uncertainty ($\sigma = 0.59$) because it is farthest from both observations.\\
450\,nm and 550\,nm have LESS uncertainty ($\sigma = 0.42$) because each is close to an observed point.}


% ============================================================
\section*{\color{secblue}Final Summary}

\begin{center}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{ccc}
\toprule
\textbf{Wavelength} & \textbf{Reflectance} & \textbf{95\% CI ($\pm 2\sigma$)} \\
\midrule
400\,nm & 0.10 & — (observed) \\
\textbf{450\,nm} & \textbf{0.1482} & $\pm$ 0.84 \\
\textbf{500\,nm} & \textbf{0.2137} & $\pm$ 1.19 \\
\textbf{550\,nm} & \textbf{0.2772} & $\pm$ 0.84 \\
600\,nm & 0.30 & — (observed) \\
\bottomrule
\end{tabular}
\end{center}

\bigskip
\note{The GP smoothly interpolated between observations, with uncertainty that is smallest near observed data and largest in the gap between them.}

\end{document}
"""

output_path = "gp_regression_worked_example.tex"
with open(output_path, "w") as f:
    f.write(latex_content)

print(f"Saved to {output_path}")
