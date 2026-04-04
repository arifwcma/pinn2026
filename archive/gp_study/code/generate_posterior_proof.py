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

\newcommand{\gut}[1]{\begin{tcolorbox}[colback=lightyellow, colframe=orange!50, title=Gut feeling]#1\end{tcolorbox}}
\newcommand{\recap}[1]{\begin{tcolorbox}[colback=lightblue, colframe=secblue!50, title=Recap]#1\end{tcolorbox}}

\pagestyle{fancy}
\fancyhf{}
\rhead{\textit{Proof of GP Posterior Formulas}}
\lhead{\textit{Arif's GP Study}}
\cfoot{\thepage}

\title{\color{secblue}\textbf{Why the GP Prediction Formulas Work}\\[0.3em]
\Large Deriving the Posterior Mean and Variance from Scratch}
\author{}
\date{}

\begin{document}
\maketitle
\thispagestyle{fancy}

% ============================================================
\section*{\color{secblue}What Are We Proving?}

In our GP worked example, we used two formulas without proof:

\begin{align}
\text{Posterior mean:} \quad \mu_{\text{pred}} &= K_{\text{pred,train}} \; K_{\text{train,train}}^{-1} \; \mathbf{y}_{\text{train}} \tag{Goal-1} \\[6pt]
\text{Posterior covariance:} \quad \Sigma_{\text{pred}} &= K_{\text{pred,pred}} - K_{\text{pred,train}} \; K_{\text{train,train}}^{-1} \; K_{\text{pred,train}}^\top \tag{Goal-2}
\end{align}

These look intimidating. But they come from one simple idea: \textbf{if you have a joint Gaussian over two groups of variables and you observe one group, the other group's distribution updates via a specific formula.}

We will derive that formula step by step, using two scalar variables first (like Step 3 of our roadmap), and then show that the GP case is the exact same thing with matrices.


% ============================================================
\section*{\color{secblue}Part 1: Setting Up the Problem}

\recap{
Remember Step 3? We had two variables $A$ and $B$, jointly Gaussian. We observed $B = 2.0$, and our belief about $A$ updated --- the mean shifted and the variance shrank. Now we prove \emph{why} it shifts and shrinks by exactly those amounts.
}

\bigskip
We have two variables with zero means (to keep things simple):

\begin{equation}
\begin{pmatrix} A \\ B \end{pmatrix}
\sim
\mathcal{N}\!\left(
\begin{pmatrix} 0 \\ 0 \end{pmatrix},\;
\begin{pmatrix} \sigma_A^2 & \sigma_{AB} \\ \sigma_{AB} & \sigma_B^2 \end{pmatrix}
\right)
\tag{P1}
\end{equation}

Let me define every symbol:

\begin{center}
\begin{tabular}{cl}
\toprule
\textbf{Symbol} & \textbf{Plain English} \\
\midrule
$A$ & The variable we want to predict (unknown) \\
$B$ & The variable we observe (known) \\
$\sigma_A^2$ & How uncertain we are about $A$ before seeing anything \\
$\sigma_B^2$ & How uncertain we are about $B$ before seeing anything \\
$\sigma_{AB}$ & How much $A$ and $B$ move together (covariance) \\
\bottomrule
\end{tabular}
\end{center}

\gut{
In our spectral GP: $A$ = reflectance at an unobserved wavelength, $B$ = reflectance at an observed wavelength, $\sigma_{AB}$ = kernel value between those two wavelengths.
}


% ============================================================
\section*{\color{secblue}Part 2: Writing the Joint PDF}

The probability density of the joint Gaussian is:

\begin{equation}
p(A, B) = \frac{1}{2\pi\,|\Sigma|^{1/2}} \;\exp\!\left( -\frac{1}{2} \;\mathbf{x}^\top \Sigma^{-1} \mathbf{x} \right)
\tag{P2}
\end{equation}

where $\mathbf{x} = \begin{pmatrix} A \\ B \end{pmatrix}$ and $\Sigma = \begin{pmatrix} \sigma_A^2 & \sigma_{AB} \\ \sigma_{AB} & \sigma_B^2 \end{pmatrix}$.

\gut{
Don't panic at this formula. It's just the bell curve from Step 1, extended to two variables. The key part is the exponent $-\frac{1}{2}\,\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$ --- this is what shapes the bell.
}


% ============================================================
\section*{\color{secblue}Part 3: Expanding the Exponent}

We already proved (Eq.~11 from the matrix inverse derivation) that:

\begin{equation}
\Sigma^{-1} = \frac{1}{\Delta}
\begin{pmatrix} \sigma_B^2 & -\sigma_{AB} \\ -\sigma_{AB} & \sigma_A^2 \end{pmatrix}
\tag{P3}
\end{equation}

where $\Delta = \sigma_A^2 \,\sigma_B^2 - \sigma_{AB}^2$ is the determinant.

\bigskip
Now let's multiply out $\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$ step by step.

\textbf{Step 3a:} Compute $\Sigma^{-1} \mathbf{x}$ (matrix times vector):

\begin{equation}
\Sigma^{-1} \begin{pmatrix} A \\ B \end{pmatrix}
= \frac{1}{\Delta}
\begin{pmatrix} \sigma_B^2 & -\sigma_{AB} \\ -\sigma_{AB} & \sigma_A^2 \end{pmatrix}
\begin{pmatrix} A \\ B \end{pmatrix}
= \frac{1}{\Delta}
\begin{pmatrix} \sigma_B^2 \,A - \sigma_{AB}\, B \\ -\sigma_{AB}\, A + \sigma_A^2\, B \end{pmatrix}
\tag{P4}
\end{equation}

\textbf{Step 3b:} Compute $\mathbf{x}^\top \cdot (\text{result from P4})$ (dot product):

\begin{equation}
\mathbf{x}^\top \Sigma^{-1} \mathbf{x}
= \frac{1}{\Delta} \Big[ A\,(\sigma_B^2\, A - \sigma_{AB}\, B) \;+\; B\,(-\sigma_{AB}\, A + \sigma_A^2\, B) \Big]
\tag{P5}
\end{equation}

\textbf{Step 3c:} Expand the brackets:

\begin{equation}
= \frac{1}{\Delta} \Big[ \;\sigma_B^2\, A^2 \;\;-\;\; \sigma_{AB}\, A\, B \;\;-\;\; \sigma_{AB}\, A\, B \;\;+\;\; \sigma_A^2\, B^2 \;\Big]
\tag{P6}
\end{equation}

\begin{equation}
= \frac{1}{\Delta} \Big[ \;\sigma_B^2\, A^2 \;\;-\;\; 2\,\sigma_{AB}\, A\, B \;\;+\;\; \sigma_A^2\, B^2 \;\Big]
\tag{P7}
\end{equation}

\gut{
Eq.~P7 is just three terms: an $A^2$ term, an $AB$ cross-term, and a $B^2$ term. Think of it like expanding $(px - qy)^2$, except it doesn't factor that neatly.
}


% ============================================================
\section*{\color{secblue}Part 4: Conditioning --- Fix $B = b$}

We observed $B = b$ (a specific number). The conditional probability is:

\begin{equation}
p(A \mid B = b) = \frac{p(A,\; B = b)}{p(B = b)}
\tag{P8}
\end{equation}

The denominator $p(B = b)$ is just a number (it doesn't depend on $A$). So:

\begin{equation}
p(A \mid B = b) \;\propto\; p(A,\; B = b)
\tag{P9}
\end{equation}

The $\propto$ means ``proportional to'' --- same shape, we just don't care about the normalizing constant because we'll recognize the shape as a Gaussian and read off the mean and variance.

\bigskip
Substitute $B = b$ into the exponent (Eq.~P7):

\begin{equation}
-\frac{1}{2\Delta} \Big[ \;\sigma_B^2\, A^2 \;\;-\;\; 2\,\sigma_{AB}\, b\, A \;\;+\;\; \underbrace{\sigma_A^2\, b^2}_{\text{constant}} \;\Big]
\tag{P10}
\end{equation}

The term $\sigma_A^2\, b^2$ does not involve $A$ at all. It's a fixed number. Since we only care about the shape as a function of $A$, we absorb it into the proportionality constant:

\begin{equation}
p(A \mid B = b) \;\propto\; \exp\!\left( -\frac{1}{2\Delta} \Big[ \;\sigma_B^2\, A^2 \;\;-\;\; 2\,\sigma_{AB}\, b\, A \;\Big] \right)
\tag{P11}
\end{equation}

\gut{
We threw away a term that doesn't depend on $A$. This is legal because we're looking at the shape of the curve in $A$. It's like saying ``$3(x-2)^2$'' and ``$7(x-2)^2$'' have the same center and width --- the constant out front doesn't change where the peak is or how wide it is.
}


% ============================================================
\section*{\color{secblue}Part 5: Completing the Square}

This is the key algebraic trick. We want to rewrite Eq.~P11 so it looks like a perfect square $(A - \text{something})^2$, because then we can immediately read off the Gaussian mean and variance.

\bigskip
\textbf{Step 5a:} Factor out $\sigma_B^2$ from inside the brackets:

\begin{equation}
\sigma_B^2\, A^2 - 2\,\sigma_{AB}\, b\, A
\;=\;
\sigma_B^2 \left( A^2 \;-\; 2\,\frac{\sigma_{AB}}{\sigma_B^2}\, b \;\cdot\; A \right)
\tag{P12}
\end{equation}

\textbf{Step 5b:} Define a shorthand: let $c = \dfrac{\sigma_{AB}}{\sigma_B^2}\, b$. Then we have:

\begin{equation}
\sigma_B^2 \left( A^2 - 2\,c\, A \right)
\tag{P13}
\end{equation}

\textbf{Step 5c:} Use the algebraic identity $A^2 - 2cA = (A - c)^2 - c^2$:

\begin{equation}
\sigma_B^2 \Big[ (A - c)^2 \;-\; c^2 \Big]
\tag{P14}
\end{equation}

The $-c^2$ part is a constant (doesn't depend on $A$), so absorb it:

\begin{equation}
p(A \mid B = b) \;\propto\; \exp\!\left( -\frac{\sigma_B^2}{2\Delta} \;(A - c)^2 \right)
\tag{P15}
\end{equation}

Substituting $c$ and $\Delta$ back:

\begin{equation}
\boxed{
p(A \mid B = b) \;\propto\; \exp\!\left( -\frac{1}{2} \;\cdot\; \frac{\left(A \;-\; \dfrac{\sigma_{AB}}{\sigma_B^2}\,b\right)^2}{\dfrac{\Delta}{\sigma_B^2}} \right)
}
\tag{P16}
\end{equation}


% ============================================================
\section*{\color{secblue}Part 6: Reading Off the Answer}

\recap{
A Gaussian $\mathcal{N}(\mu,\, v)$ has the PDF shape $\exp\!\Big(-\frac{1}{2}\,\frac{(x - \mu)^2}{v}\Big)$.

So whatever sits inside the $(x - \boxed{\phantom{x}})^2$ is the \textbf{mean}, and whatever sits in the denominator is the \textbf{variance}.
}

\bigskip
Comparing Eq.~P16 with the Gaussian template:

\begin{equation}
\boxed{\text{Conditional mean} = \frac{\sigma_{AB}}{\sigma_B^2} \;\cdot\; b}
\tag{P17}
\end{equation}

``Covariance between $A$ and $B$, divided by variance of $B$, times the observed value of $B$.''

\gut{
If $A$ and $B$ are strongly correlated ($\sigma_{AB}$ large), the mean shifts a lot toward what $B$ suggests. If they're uncorrelated ($\sigma_{AB} = 0$), the mean stays at 0 --- observing $B$ told us nothing.
}

\begin{equation}
\boxed{\text{Conditional variance} = \frac{\Delta}{\sigma_B^2} = \sigma_A^2 - \frac{\sigma_{AB}^2}{\sigma_B^2}}
\tag{P18}
\end{equation}

``Prior variance of $A$, minus the squared-correlation divided by variance of $B$.''

\gut{
The variance always \emph{shrinks} (or stays the same). The subtracted term $\sigma_{AB}^2 / \sigma_B^2$ is always $\geq 0$, so observing $B$ can never make you \emph{more} uncertain about $A$. This is why we said in R9: ``data can never hurt.''
}


% ============================================================
\section*{\color{secblue}Part 7: Verify with Step 3 Numbers}

From our earlier example: $\sigma_A^2 = 1$, $\sigma_B^2 = 1$, $\sigma_{AB} = 0.8$, $b = 2.0$.

\begin{align}
\text{Mean} &= \frac{0.8}{1.0} \times 2.0 = 1.6 \quad \checkmark \tag{P19} \\[8pt]
\text{Variance} &= 1.0 - \frac{0.8^2}{1.0} = 1.0 - 0.64 = 0.36 \quad \checkmark \tag{P20}
\end{align}

Exactly what our code produced in Step 3.


% ============================================================
\section*{\color{secblue}Part 8: From Scalars to GP --- The Jump to Matrices}

Everything above used two scalar variables. In GP, we have \textbf{vectors} of variables:

\begin{itemize}
\item $A$ becomes $\mathbf{f}_{\text{pred}}$ --- a vector of unknown reflectances (e.g., at 450, 500, 550\,nm)
\item $B$ becomes $\mathbf{f}_{\text{train}}$ --- a vector of observed reflectances (e.g., at 400, 600\,nm)
\end{itemize}

\bigskip
Every scalar quantity becomes a matrix:

\begin{center}
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{ccc}
\toprule
\textbf{Scalar (2 variables)} & \textbf{becomes} & \textbf{GP (vectors of variables)} \\
\midrule
$\sigma_A^2$ (variance of $A$) & $\longrightarrow$ & $K_{\text{pred,pred}}$ (covariance among prediction points) \\
$\sigma_B^2$ (variance of $B$) & $\longrightarrow$ & $K_{\text{train,train}}$ (covariance among observed points) \\
$\sigma_{AB}$ (covariance) & $\longrightarrow$ & $K_{\text{pred,train}}$ (cross-covariance) \\
$b$ (observed value) & $\longrightarrow$ & $\mathbf{y}_{\text{train}}$ (observed reflectances) \\
dividing by $\sigma_B^2$ & $\longrightarrow$ & multiplying by $K_{\text{train,train}}^{-1}$ \\
\bottomrule
\end{tabular}
\end{center}

\bigskip
Making these substitutions in Eqs.~P17 and P18:

\begin{equation}
\boxed{\boldsymbol{\mu}_{\text{pred}} = K_{\text{pred,train}} \;\; K_{\text{train,train}}^{-1} \;\; \mathbf{y}_{\text{train}}}
\tag{P21}
\end{equation}

Compare with Eq.~P17: $\dfrac{\sigma_{AB}}{\sigma_B^2} \cdot b$ became $K_{\text{pred,train}} \cdot K_{\text{train,train}}^{-1} \cdot \mathbf{y}_{\text{train}}$.

\begin{equation}
\boxed{\Sigma_{\text{pred}} = K_{\text{pred,pred}} \;-\; K_{\text{pred,train}} \;\; K_{\text{train,train}}^{-1} \;\; K_{\text{pred,train}}^\top}
\tag{P22}
\end{equation}

Compare with Eq.~P18: $\sigma_A^2 - \dfrac{\sigma_{AB}^2}{\sigma_B^2}$ became $K_{\text{pred,pred}} - K_{\text{pred,train}} \cdot K_{\text{train,train}}^{-1} \cdot K_{\text{pred,train}}^\top$.

\gut{
The $\sigma_{AB}^2$ in the scalar case became $K_{\text{pred,train}} \cdot K_{\text{pred,train}}^\top$ in the matrix case. That's because for scalars, $\sigma_{AB} \cdot \sigma_{AB} = \sigma_{AB}^2$. For matrices, you can't square a non-square matrix, so instead you multiply by its transpose: $K_{\text{pred,train}} \cdot K_{\text{pred,train}}^\top$.
}

\bigskip
The proof in the matrix case follows the identical logic --- write the joint PDF, fix the observed variables, complete the square in the unknown variables, read off mean and covariance. The algebra is messier but the structure is the same.


% ============================================================
\section*{\color{secblue}Summary: The Whole Proof in 5 Lines}

\begin{enumerate}
\item We wrote the joint Gaussian PDF over $(A, B)$ \hfill (Eq.~P2)
\item We expanded the exponent into $A^2$, $AB$, and $B^2$ terms \hfill (Eq.~P7)
\item We fixed $B = b$ and dropped terms that don't involve $A$ \hfill (Eq.~P11)
\item We completed the square to get $(A - \text{mean})^2 / \text{variance}$ \hfill (Eq.~P16)
\item We read off the mean and variance from the Gaussian shape \hfill (Eqs.~P17, P18)
\end{enumerate}

Then we replaced scalars with matrices to get the GP formulas (Eqs.~P21, P22). \qed

\end{document}
"""

output_path = "gp_posterior_proof.tex"
with open(output_path, "w") as f:
    f.write(latex_content)

print(f"Saved to {output_path}")
