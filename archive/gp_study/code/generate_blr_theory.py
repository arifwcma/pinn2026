latex_content = r"""
\documentclass[11pt, a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{tcolorbox}
\setlength{\headheight}{14pt}

\definecolor{secblue}{RGB}{26, 71, 111}
\definecolor{notegreen}{RGB}{40, 120, 60}
\definecolor{explgray}{RGB}{100, 100, 100}
\definecolor{lightblue}{RGB}{230, 242, 255}
\definecolor{lightyellow}{RGB}{255, 250, 230}
\definecolor{lightgreen}{RGB}{230, 250, 230}

\newcommand{\gut}[1]{\begin{tcolorbox}[colback=lightyellow, colframe=orange!50, title=Gut feeling]#1\end{tcolorbox}}
\newcommand{\recap}[1]{\begin{tcolorbox}[colback=lightblue, colframe=secblue!50, title=Recap]#1\end{tcolorbox}}
\newcommand{\bridge}[1]{\begin{tcolorbox}[colback=lightgreen, colframe=green!40, title=Bridge to GP]#1\end{tcolorbox}}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textit{Bayesian Linear Regression}}
\lhead{\textit{Arif's GP Study}}
\cfoot{\thepage}

\title{\color{secblue}\textbf{Bayesian Linear Regression}\\[0.3em]
\Large From OLS to Uncertainty --- The Bridge to Gaussian Processes}
\author{}
\date{}

\begin{document}
\maketitle
\thispagestyle{fancy}


% ============================================================
\section*{\color{secblue}How BLR Differs from OLS}

\begin{center}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{lcc}
\toprule
& \textbf{OLS} & \textbf{Bayesian LR} \\
\midrule
Weights $\mathbf{w}$ are\ldots & fixed unknown numbers & \textbf{random variables} with a distribution \\
Before data & no belief about $\mathbf{w}$ & prior: $p(\mathbf{w})$ \\
After data & one best $\mathbf{w}$ & posterior: $p(\mathbf{w} \mid \text{data})$ \\
Prediction & one number & \textbf{mean $\pm$ uncertainty} \\
\bottomrule
\end{tabular}
\end{center}

\gut{
In OLS, the weights are treated as fixed but unknown constants that we estimate.
In BLR, the weights are treated as random variables --- they have a probability distribution that expresses our uncertainty about their true values. Seeing data updates this distribution.
}


% ============================================================
\section*{\color{secblue}The Setup}

Same data as OLS: $n$ houses, each with size $x_i$ and price $y_i$.
Same model:

\begin{equation}
y_i = w_0 + w_1 \, x_i + \epsilon_i
\tag{B-1}
\end{equation}

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Plain English} \\
\midrule
$w_0, w_1$ & Intercept and slope (now random variables, not fixed) \\
$\epsilon_i$ & Noise --- the random error in each measurement \\
$\sigma_n^2$ & How much noise we expect (known or estimated) \\
\bottomrule
\end{tabular}
\end{center}

The noise is Gaussian: $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$.

This means each observation is:

\begin{equation}
y_i \mid \mathbf{w} \sim \mathcal{N}(w_0 + w_1 x_i, \;\; \sigma_n^2)
\tag{B-2}
\end{equation}

``Given the weights, each price is Gaussian-distributed around the line, with spread $\sigma_n$.''


% ============================================================
\section*{\color{secblue}Step 1: The Prior --- Our Belief Before Seeing Data}

Before looking at any houses, we state our belief about $\mathbf{w} = (w_0, w_1)^\top$:

\begin{equation}
\mathbf{w} \sim \mathcal{N}(\mathbf{m}_0, \;\; S_0)
\tag{B-3}
\end{equation}

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Plain English} \\
\midrule
$\mathbf{m}_0$ & Our initial guess for $[w_0, w_1]$ (often just $[0, 0]$ --- ``no idea'') \\
$S_0$ & How uncertain we are about that guess (a $2\times 2$ covariance matrix) \\
\bottomrule
\end{tabular}
\end{center}

\gut{
A wide $S_0$ (large values on the diagonal) means ``I really don't know what the slope or intercept is --- many values are plausible.'' A narrow $S_0$ means ``I'm fairly confident they're near $\mathbf{m}_0$.'' Setting $\mathbf{m}_0 = [0, 0]$ with large $S_0$ is the humble starting point: ``I have no idea, any line is possible.''
}

For our house example, let's use:

\begin{equation}
\mathbf{m}_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad
S_0 = \begin{pmatrix} 100 & 0 \\ 0 & 100 \end{pmatrix}
\tag{B-4}
\end{equation}

This says: ``The intercept is probably somewhere between $-20$ and $+20$ (within $\pm 2\sigma = \pm 2\sqrt{100}$), and same for the slope. I'm not committing to anything.''


% ============================================================
\section*{\color{secblue}Step 2: The Likelihood --- What the Data Tells Us}

The likelihood answers: ``Given a specific line (specific $\mathbf{w}$), how probable is the data we observed?''

For all $n$ data points together (in matrix form):

\begin{equation}
p(\mathbf{y} \mid \mathbf{w}) = \mathcal{N}(X\mathbf{w}, \;\; \sigma_n^2 I)
\tag{B-5}
\end{equation}

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Plain English} \\
\midrule
$X\mathbf{w}$ & The predictions this line would make for all houses \\
$\sigma_n^2 I$ & Each observation has independent noise $\sigma_n^2$ \\
$I$ & Identity matrix (noise is independent between samples) \\
\bottomrule
\end{tabular}
\end{center}

\gut{
If a candidate line passes close to all data points, the likelihood is high --- the data is ``probable'' under that line. If the line misses badly, the likelihood is low. The likelihood is what connects the abstract prior belief to the concrete data.
}


% ============================================================
\section*{\color{secblue}Step 3: The Posterior --- Belief After Seeing Data}

Bayes' theorem combines prior and likelihood:

\begin{equation}
\underbrace{p(\mathbf{w} \mid \mathbf{y})}_{\text{posterior}} \;\propto\; \underbrace{p(\mathbf{y} \mid \mathbf{w})}_{\text{likelihood}} \;\cdot\; \underbrace{p(\mathbf{w})}_{\text{prior}}
\tag{B-6}
\end{equation}

``Posterior = how well the line fits the data $\times$ how plausible the line was before seeing data.''

\gut{
A line that fits the data brilliantly but was extremely unlikely a priori will get moderate posterior probability. A line that was very plausible a priori but fits the data poorly will also get moderate posterior probability. The posterior rewards lines that are \textbf{both} plausible and consistent with data.
}

Because both the prior (Eq.~B-3) and likelihood (Eq.~B-5) are Gaussian, the posterior is also Gaussian (this is the ``conjugacy'' property --- Gaussian $\times$ Gaussian = Gaussian):

\begin{equation}
\boxed{p(\mathbf{w} \mid \mathbf{y}) = \mathcal{N}(\mathbf{m}_n, \;\; S_n)}
\tag{B-7}
\end{equation}

with:

\begin{equation}
\boxed{S_n = \left( S_0^{-1} + \frac{1}{\sigma_n^2} X^\top X \right)^{-1}}
\tag{B-8}
\end{equation}

\begin{equation}
\boxed{\mathbf{m}_n = S_n \left( S_0^{-1} \mathbf{m}_0 + \frac{1}{\sigma_n^2} X^\top \mathbf{y} \right)}
\tag{B-9}
\end{equation}

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Plain English} \\
\midrule
$\mathbf{m}_n$ & Our updated best guess for the weights (posterior mean) \\
$S_n$ & Our updated uncertainty about the weights (posterior covariance) \\
$S_0^{-1}$ & How confident we were before (inverse prior covariance) \\
$\frac{1}{\sigma_n^2} X^\top X$ & How much information the data provides \\
$\frac{1}{\sigma_n^2} X^\top \mathbf{y}$ & What the data suggests the weights should be \\
\bottomrule
\end{tabular}
\end{center}

\gut{
Look at Eq.~B-8 carefully. The posterior precision (inverse of uncertainty) is:
$$S_n^{-1} = \underbrace{S_0^{-1}}_{\text{prior confidence}} + \underbrace{\frac{1}{\sigma_n^2} X^\top X}_{\text{data's contribution}}$$
More data or less noise $\rightarrow$ larger second term $\rightarrow$ higher precision $\rightarrow$ less uncertainty. The posterior is \textbf{always at least as certain as the prior}. Sound familiar? This is exactly the same principle as GP's ``data can never hurt'' from R9.
}


% ============================================================
\section*{\color{secblue}Step 4: Making Predictions}

For a new house with size $x_*$, we build $\mathbf{x}_* = (1, \; x_*)^\top$ and predict:

\begin{equation}
\boxed{p(y_* \mid \mathbf{y}) = \mathcal{N}\!\left(\mathbf{x}_*^\top \mathbf{m}_n, \;\; \mathbf{x}_*^\top S_n \, \mathbf{x}_* + \sigma_n^2\right)}
\tag{B-10}
\end{equation}

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Component} & \textbf{Plain English} \\
\midrule
$\mathbf{x}_*^\top \mathbf{m}_n$ & Predicted price (plug $x_*$ into the best-fit line) \\
$\mathbf{x}_*^\top S_n \, \mathbf{x}_*$ & Uncertainty from not knowing the exact weights \\
$\sigma_n^2$ & Uncertainty from measurement noise \\
\bottomrule
\end{tabular}
\end{center}

\gut{
The prediction uncertainty has \textbf{two sources}:
\begin{enumerate}
\item \textbf{Epistemic} ($\mathbf{x}_*^\top S_n \, \mathbf{x}_*$): ``I'm not sure which line is correct.'' This shrinks as you get more data.
\item \textbf{Aleatoric} ($\sigma_n^2$): ``Even if I knew the perfect line, individual houses vary.'' This never shrinks --- it's inherent randomness.
\end{enumerate}
OLS gives you neither. BLR gives you both. This is the key upgrade.
}


% ============================================================
\section*{\color{secblue}Connection to OLS}

If the prior is extremely vague ($S_0 \rightarrow \infty I$, meaning $S_0^{-1} \rightarrow 0$), Eqs.~B-8 and B-9 simplify:

\begin{align}
S_n &\approx \sigma_n^2 (X^\top X)^{-1} \tag{B-11} \\
\mathbf{m}_n &\approx (X^\top X)^{-1} X^\top \mathbf{y} \tag{B-12}
\end{align}

Eq.~B-12 is exactly the OLS solution (Eq.~OLS-14). So \textbf{OLS is a special case of BLR} --- the case where the prior carries no information.

\bridge{
In BLR, we have uncertainty over 2 parameters (slope, intercept) $\rightarrow$ uncertainty in predictions.

In GP, we have uncertainty over \textbf{infinitely many} function values $\rightarrow$ uncertainty in predictions.

The mechanism is identical: prior $\rightarrow$ observe data $\rightarrow$ posterior (conditioning). GP is the limit of BLR as the number of parameters goes to infinity.
}


% ============================================================
\section*{\color{secblue}Summary: The Three Ingredients}

\begin{enumerate}
\item \textbf{Prior} $p(\mathbf{w}) = \mathcal{N}(\mathbf{m}_0, S_0)$ --- what you believe before data \hfill (Eq.~B-3)
\item \textbf{Likelihood} $p(\mathbf{y} \mid \mathbf{w}) = \mathcal{N}(X\mathbf{w}, \sigma_n^2 I)$ --- what the data says \hfill (Eq.~B-5)
\item \textbf{Posterior} $p(\mathbf{w} \mid \mathbf{y}) = \mathcal{N}(\mathbf{m}_n, S_n)$ --- your updated belief \hfill (Eq.~B-7)
\end{enumerate}

The posterior mean $\mathbf{m}_n$ gives you the best line. The posterior covariance $S_n$ tells you how much to trust it. Together, they give predictions with honest uncertainty.

\end{document}
"""

output_path = "bayesian_linear_regression.tex"
with open(output_path, "w") as f:
    f.write(latex_content)

print(f"Saved to {output_path}")
