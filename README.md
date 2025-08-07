# Variational Monte Carlo with Neural Network States

In this tutorial we will explore neural network wavefunction Ansätze to perform variational optimization of the energy expectation value in order to approximate quantum mechanical ground states.

## Energy expectation of quantum systems

Consider a quantum state expressed in a complete basis $\{\ket{\boldsymbol\sigma}\}$, i.e. $\ket\psi = \sum_{\boldsymbol\sigma}\ket{\boldsymbol\sigma}\braket{\boldsymbol\sigma|\psi} = \sum_{\boldsymbol\sigma} \psi(\boldsymbol\sigma) \ket{\boldsymbol\sigma}$ with wavefunction $\psi(\boldsymbol\sigma)=\braket{\boldsymbol\sigma|\psi}$.
The energy expectation value in the basis of configurations $\boldsymbol\sigma$ is given by

```math
    \begin{align*}
        E
        &=
        \frac{\braket{\psi | \hat H | \psi}}{\braket{\psi | \psi}}
        =
        \sum_{\boldsymbol\sigma,\boldsymbol\sigma'}
        \frac{\braket{\psi \ket{\boldsymbol\sigma}\bra{\boldsymbol\sigma} \hat H \ket{\boldsymbol\sigma'}\bra{\boldsymbol\sigma'} \psi}}{\braket{\psi|\psi}}
        =
        \sum_{\boldsymbol\sigma,\boldsymbol\sigma'}
        \braket{\boldsymbol\sigma | \hat H | \boldsymbol\sigma'}\frac{\psi^*(\boldsymbol\sigma)\psi(\boldsymbol\sigma')}{\braket{\psi | \psi}}
        \\
        &=
        \sum_{\boldsymbol\sigma}
        \frac{|\psi(\boldsymbol\sigma)|^2}{\braket{\psi | \psi}}
        \sum_{\boldsymbol\sigma'}
        \braket{\boldsymbol\sigma | \hat H | \boldsymbol\sigma'}
        \frac{\psi(\boldsymbol\sigma')}{\psi(\boldsymbol\sigma)}
        \\
        % &=
        % \sum_{\boldsymbol\sigma,\boldsymbol\sigma'}
        % p(\boldsymbol\sigma)
        % \braket{\boldsymbol\sigma | \hat H | \boldsymbol\sigma'}
        % \frac{\psi(\boldsymbol\sigma')}{\psi(\boldsymbol\sigma)}
        % \\
        &=
        \sum_{\boldsymbol\sigma}
        p(\boldsymbol\sigma)
        E_{\rm loc}(\boldsymbol\sigma)
    \end{align*}
```

where in the second line we inserted $\psi(\boldsymbol\sigma)/\psi(\boldsymbol\sigma)$, and in the last line defined the probability amplitude $p(\boldsymbol\sigma) = |\psi(\boldsymbol\sigma)|^2/\braket{\psi|\psi}$ and local energy $E_{\rm loc}(\boldsymbol\sigma) = \sum_{\boldsymbol\sigma'} \braket{\boldsymbol\sigma | \hat H | \boldsymbol\sigma'}\frac{\psi(\boldsymbol\sigma')}{\psi(\boldsymbol\sigma)}$.
Using samples $\boldsymbol\sigma_i$ appropriately drawn from the probability distribution $p(\boldsymbol\sigma)$, we may approximate the sum according to

```math
    E \approx \frac1{n_s}\sum_{i=1}^{n_s}E_{\rm loc}(\boldsymbol\sigma_i)
    .
```

## Importance Sampling

In case the probability density $p(\boldsymbol\sigma)$ is dominated by a small part of the whole state space (such as for ground states), Markov Chain Monte Carlo (MCMC) is an efficient approach to perform the sampling.

Starting from an initial state, samples are drawn iteratively and form a Markov Chain.
In order to perform expectation values, the Markov Chain must converge to the (stationary) distribution $p(\boldsymbol\sigma)$ regardless of the initial choice.

Define $t(\boldsymbol\sigma\rightarrow\boldsymbol\sigma')$ to be the (normalized) transition probability from the configuration $\boldsymbol\sigma$ to $\boldsymbol\sigma'$ (such that $\sum_{\boldsymbol\sigma'} t(\boldsymbol\sigma\rightarrow\boldsymbol\sigma') = 1$), and $p_s(\boldsymbol\sigma)$ the probability to be in the state $\boldsymbol\sigma$ at step $s$, then

```math
    p_{s+1}(\boldsymbol\sigma) = \sum_{\boldsymbol\sigma'} p_s(\boldsymbol\sigma') t(\boldsymbol\sigma'\rightarrow\boldsymbol\sigma)
    .
```

A stationary distribution is obtained when $p_s$ is independent on $s$.
If a Markov chain is irreducible (reaches any state from any other state in a finite number of steps with positive probability), then the stationary distribution is unique, and if the chain is further aperiodic (in the discrete case, $t(\boldsymbol\sigma\rightarrow\boldsymbol\sigma)>0$), the step-dependent probability converges to it.

A sufficient but not necessary condition for stationarity is detailed balance, which states that transitions must be reversible:

```math
    p(\boldsymbol\sigma) t(\boldsymbol\sigma \rightarrow \boldsymbol\sigma') = p(\boldsymbol\sigma') t(\boldsymbol\sigma' \rightarrow \boldsymbol\sigma).
```

## The Metropolis-Hastings algorithm

The Metropolis-Hastings algorithm satisfies detailed balance, where the transition amplitude is split into two parts, the probability to draw and accept the next state

```math
    t(\boldsymbol\sigma \rightarrow \boldsymbol\sigma') = t_{\rm accept}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')\, t_{\rm next}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')
    .
```

The algorithm picks an acceptance probability
```math
    t_{\rm accept}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')={\rm min}\left\{1, \frac{p(\boldsymbol\sigma')}{p(\boldsymbol\sigma)}\frac{t_{\rm next}(\boldsymbol\sigma' \rightarrow \boldsymbol\sigma)}{t_{\rm next}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')}\right\}
```

 which is further simplified for in the symmetric case $t_{\rm next}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma') = t_{\rm next}(\boldsymbol\sigma' \rightarrow \boldsymbol\sigma)$ to

```math
    t_{\rm accept}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')
    =
    {\rm min}\left\{1, \frac{p(\boldsymbol\sigma')}{p(\boldsymbol\sigma)}\right\}
    .
```

We can explicitly notice that detailed balance is fulfilled by inspecting the two equations

```math
    \begin{align*}
        p(\boldsymbol\sigma) t(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')
        &=
        p(\boldsymbol\sigma) t_{\rm accept}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')
        =
        p(\boldsymbol\sigma)\ {\rm min}\left\{1, \frac{p(\boldsymbol\sigma')}{p(\boldsymbol\sigma)}\right\}
        =
        \left\{
            \begin{array}{c}
                p(\boldsymbol\sigma)\text{ for } p(\boldsymbol\sigma') \geq p(\boldsymbol\sigma)\\
                p(\boldsymbol\sigma')\text{ for } p(\boldsymbol\sigma') < p(\boldsymbol\sigma)\\
            \end{array}
        \right.
        ,
        \\
        p(\boldsymbol\sigma') t(\boldsymbol\sigma' \rightarrow \boldsymbol\sigma)
        &=
        p(\boldsymbol\sigma')\ {\rm min}\left\{1, \frac{p(\boldsymbol\sigma)}{p(\boldsymbol\sigma')}\right\}
        =
        \left\{
            \begin{array}{c}
                p(\boldsymbol\sigma)\text{ for } p(\boldsymbol\sigma') \geq p(\boldsymbol\sigma)\\
                p(\boldsymbol\sigma')\text{ for } p(\boldsymbol\sigma') < p(\boldsymbol\sigma)\\
            \end{array}
        \right.
        .
    \end{align*}
```

The Markov Chain is generated by iterating over two steps:

1) Given a configuration $\boldsymbol\sigma$, draw a new configuration $\boldsymbol\sigma'$ with probability $t_{\rm next}(\boldsymbol\sigma \rightarrow \boldsymbol\sigma')$.
2) Accept the new configuration with probability $t_{\rm accept}$. If rejected, continue with $\boldsymbol\sigma$.

The number of iterations to reach a stationary distribution is called thermalisation or burn-in phase.
Colloquially speaking, it is the time it takes the Markov Chain to forget about the initial state.
Only after the thermalisation process, it makes sense to draw samples from the distribution to compute expectation values.
However, samples drawn at successive Metropolis-Hastings iterations are in general highly correlated, and therefore not correctly reflect the stationary distribution.
It is thus necessary to compute the 'autocorrelation time' $n_{at}$ (estimate after how many iterations samples are independent), and draw samples forming a true Markov Chain after $n_{at}$ successive iterations.

## Autocorrelation time

In order to estimate correlations between samples, we compute the [autocorrelation and efficiency](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo#:~:text=Autocorrelation%20and%20efficiency).
The integrated autocorrelation is defined by

```math
    R = 1 + 2\sum_k\rho_k
```

where $\rho_k = {\rm Cov}(X_0, X_k) / \sqrt{{\rm Var}(X_0){\rm Var}(X_k)}$ is the autocorrelation.
In practice, it is an estimate for the fraction of the effective size of independent samples in a correlated set, i.e. $N_{\rm eff} = N / R$.

## Restricted Boltzmann Machine (Rboldsymbol) Ansatz

We represent the wave function $\psi(\boldsymbol\sigma)$ by the following parametrized function

```math
    \psi_{\boldsymbol\theta}(\boldsymbol\sigma)
    =
    {\rm e}^{\boldsymbol b\cdot\boldsymbol\sigma}\prod_{i=1}^{n_h}2\cosh(\varphi_i(\boldsymbol\sigma))
    ,\
    \boldsymbol\varphi(\boldsymbol\sigma) = \boldsymbol c + W\boldsymbol\sigma
```

for some suitable choice of the $n_p$ variational parameters $\boldsymbol\theta_i\in\{\boldsymbol b, \boldsymbol c, W\}$, where $\boldsymbol b\in \mathbb C^{n_{\rm spins}}$, $\boldsymbol c\in \mathbb C^{n_h}$ are complex vectors and $W$ is a complex $n_h\times n_{\rm spins}$ matrix.

See [Carleo & Troyer](https://www.science.org/doi/10.1126/science.aag2302) for a detailed discussion of this Ansatz.

## Parameter Updates and Stochastic Reconfiguration (SR)

In order to perform an optimal parameter update, we seek for an optimal approximation of the imaginary time-evolved state

```math
    \ket{\psi_{\boldsymbol\theta}'}
    =
    {\rm e}^{-\epsilon\hat H}\ket{\psi_{\boldsymbol\theta}}
    \approx
    \ket{\psi_{\boldsymbol\theta}} - \epsilon\hat H \ket{\psi_{\boldsymbol\theta}}
    =
    \lambda_0\ket{\psi_{\boldsymbol\theta}}
    +
    \lambda_i\partial_{\theta_i}\ket{\psi_{\boldsymbol\theta}}
    +
    \lambda_\perp \ket{\psi_{\perp}}
    \\
    =
    \lambda_0
    \left(
        \ket{\psi_{\boldsymbol\theta}}
        +
        \lambda_i/\lambda_0\partial_{\theta_i}\ket{\psi_{\boldsymbol\theta}}
    \right)
    +
    \lambda_\perp \ket{\psi_{\perp}}
    =
    \lambda_0
    \ket{\psi_{\boldsymbol\theta + \delta\boldsymbol\theta}}
    +
    \lambda_\perp \ket{\psi_{\perp}}
```

where we expanded the time-evolved state in the subspace spanned by $\{\ket{\psi_{\boldsymbol\theta}}, \partial_{\theta_i}\ket{\psi_{\boldsymbol\theta}}\}$, and an orthogonal state.
We recognize that the first two terms can be interpreted as a Taylor approximation of a state with slightly adapted variational parameters $\delta\theta_i = \lambda_i/\lambda_0$.
By projecting onto $\bra{\psi_{\boldsymbol\theta}}$ and $\bra{\psi_{\boldsymbol\theta}}\partial_{\theta_j}^\dagger$, (assuming $\ket{\psi_{\boldsymbol\theta}}$ is normalized), we find the two equations

```math
\begin{align*}
    1 - \epsilon\braket{\hat H}
    &=
    \lambda_0 + \lambda_i\braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i} | \psi_{\boldsymbol\theta}}
    ,
    \\
    \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger | \psi_{\boldsymbol\theta}} - \epsilon \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger \hat H | \psi_{\boldsymbol\theta}}
    &=
    \lambda_0 \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger | \psi_{\boldsymbol\theta}} + \lambda_i \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger \partial_{\theta_i} | \psi_{\boldsymbol\theta}}
\end{align*}
```

which can be combined by eliminating $\lambda_0 = 1 - \epsilon\braket{\hat H} - \lambda_i \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i} | \psi_{\boldsymbol\theta}}$ and swapping the indices $i\leftrightarrow j$, resulting in

```math
    \lambda_j
    \left(
        \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger \partial_{\theta_j} | \psi_{\boldsymbol\theta}}
        -
        \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger | \psi_{\boldsymbol\theta}}
        \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j} | \psi_{\boldsymbol\theta}}
    \right)
    =
    -\epsilon
    \left(
        \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger \hat H | \psi_{\boldsymbol\theta}}
        -
        \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger | \psi_{\boldsymbol \theta}}
        \braket{\psi_{\boldsymbol\theta} | \hat H | \psi_{\boldsymbol \theta}}
    \right)
    .
```

Using the quantum metric tensor $S_{ij} = \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger \partial_{\theta_j} | \psi_{\boldsymbol\theta}} - \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_i}^\dagger | \psi_{\boldsymbol\theta}}\braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j} | \psi_{\boldsymbol\theta}}$ and the "effective forces" $R_j = \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger \hat H | \psi_{\boldsymbol\theta}} - \braket{\psi_{\boldsymbol\theta} | \partial_{\theta_j}^\dagger | \psi_{\boldsymbol \theta}} \braket{\psi_{\boldsymbol\theta} | \hat H | \psi_{\boldsymbol \theta}}$, the previous equation can be recast into

```math
    \delta\theta_i = -\eta S^{-1}_{ij} R_j
```

where $\eta = \epsilon / \lambda_0$ is the learning rate.
Note that by imposing an imaginary learning rate $\eta\rightarrow{\rm i}\eta$, we perform real time evolution.

Due to the simplicity of the Ansatz, we can express the wave function derivatives:

```math
\begin{align*}
    \braket{\psi_{\boldsymbol\theta} | \partial_{b_i} | \psi_{\boldsymbol\theta}}
    &=
    \sum_{\sigma} p(\sigma)\sigma_i
    ,
    \\
    \braket{\psi_{\boldsymbol\theta} | \partial_{c_i} | \psi_{\boldsymbol\theta}}
    &=
    \sum_{\sigma} p(\sigma)\tanh(\varphi_i)
    ,
    \\
    \braket{\psi_{\boldsymbol\theta} | \partial_{W_{ij}} | \psi_{\boldsymbol\theta}}
    &=
    \sum_{\sigma} p(\sigma)\sigma_j\tanh(\varphi_i)
    .
\end{align*}
```

## From SR to [MinSR](https://www.nature.com/articles/s41567-024-02566-1)

We employ the singular value decomposition $\overline O = U \Lambda V^\dagger$ to establish the equivalence between

```math
    (\overline O^\dagger \overline O)^{-1} \overline O^\dagger
    =
    (V \Lambda^2 V^\dagger)^{-1}
    V \Lambda U^\dagger
    =
    V\Lambda^{-1} U^\dagger
```

and

```math
    \overline O^\dagger(\overline O \overline O^\dagger)^{-1}
    =
    V \Lambda U^\dagger (U \Lambda^2 U^\dagger)^{-1}
    =
    V\Lambda^{-1} U^\dagger
    .
```

Define $\overline O_{\boldsymbol\sigma,j} = (O_{\boldsymbol\sigma} - \braket{\partial_{\theta_j}})/\sqrt{n_s}$ with $O_{\boldsymbol\sigma} = \partial_{\boldsymbol\theta}\ln\psi_{\boldsymbol\theta}(\boldsymbol\sigma)$, $\overline H_{\boldsymbol\sigma} = (E_{\rm loc}(\boldsymbol\sigma) - \braket{\hat H})/\sqrt{n_s}$, and $\overline O_{ij}$ (similarly $\overline H$) as the matrix element of the sampled value $\overline O_{\boldsymbol\sigma_i,j}$, then we can define the SR update as

```math
    \delta\boldsymbol\theta
    =
    -\eta (\overline O^\dagger\,\overline O)^{-1}\overline O^\dagger \overline H
    .
```

Since $\overline O$ is a matrix of dimensions $n_s \times n_p$, where $n_p$ is the number of variational parameters, SR requires to evaluate and invert the $n_p\times n_p$ matrix $S$.
MinSR, on the other hand, reads

```math
    \delta\boldsymbol\theta
    =
    -\eta \overline O^\dagger(\overline O\,\overline O^\dagger)^{-1} \overline H
    ,
```

and requires to invert the $n_s \times n_s$ matrix $T = \overline O\,\overline O^\dagger$.
We can easily notice the computational advantage of MinSR, because an expressive wavefunction Ansatz satisfies $n_s\ll n_p$.
