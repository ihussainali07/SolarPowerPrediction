# Task 5 – Analysis

## 5.1 Normal Equation and Feature Weights — Set A

The Normal Equation was used to learn the parameter vector θ for Set A. The learned parameters are shown below.

| Feature | Normal Equation θ |
|---|---:|
| Intercept | 6803.188733 |
| Irradiation | 8187.254344 |
| Module temperature | 5.329879 |
| Ambient temperature | -65.130812 |
| sin_hour | -86.115151 |
| cos_hour | -606.321145 |

The largest absolute non-intercept weight is for **irradiation**, with a value of **8187.254344**. Therefore, irradiation has the largest learned coefficient among the input features.

The signs of the coefficients can be interpreted using the expected physical behavior of a photovoltaic system.

- **Irradiation: +8187.254344** — The positive sign agrees with the physics. Higher solar irradiation provides more available solar energy and should generally increase PV power output.
- **Module temperature: +5.329879** — The coefficient is positive, which does not directly agree with the usual physical expectation that increasing PV module temperature reduces electrical efficiency. However, the coefficient is very small compared with the irradiation coefficient, and the model contains correlated environmental variables. Therefore, this coefficient should not be interpreted in isolation.
- **Ambient temperature: -65.130812** — The negative sign can be consistent with the indirect relationship between temperature and PV output, but ambient temperature is correlated with module temperature and irradiation. Therefore, its coefficient does not represent a simple one-variable physical effect.
- **sin_hour: -86.115151** and **cos_hour: -606.321145** — These are engineered time-of-day features. Their individual signs do not have a direct physical interpretation like irradiation or module temperature. Together, they model the periodic daily pattern of solar generation.

Overall, the irradiation coefficient has the strongest learned effect. The irradiation sign agrees clearly with the expected PV physics. The module-temperature coefficient has a positive sign rather than the commonly expected negative sign, so that part of the learned linear model does not directly agree with the expected physical relationship. The ambient-temperature and time-of-day coefficients require more careful interpretation because of correlations and feature engineering.

## 5.2 Set B vs Set A Daytime RMSE

The assignment requires comparing the daytime RMSE of Set B against Set A, both in kW and as a percentage of the plant's peak hourly power.

**The numerical values for this section are not included in the supplied Task 5.1 θ table, so they should be taken from the program's Set A/Set B RMSE results rather than estimated here.**

Use the following calculation:

- Set A daytime RMSE = `RMSE_A_day`
- Set B daytime RMSE = `RMSE_B_day`
- Difference = `RMSE_B_day - RMSE_A_day`
- Percentage of plant peak hourly power =

\[
100 \times \frac{RMSE_B\_day - RMSE_A\_day}{Peak\ hourly\ power}
\]

The interpretation should state whether the additional error from using public weather data is small or large relative to the plant's peak output.

For the rooftop-installer question, the conclusion should be based on the measured percentage: if the additional error is small enough for the intended planning/application, public weather data may be useful as a low-cost alternative; if accurate site-specific prediction is required, on-site measurements are preferable.

## 5.3 Comparison of the Three Solvers

Three solvers were used: the Normal Equation, Batch Gradient Descent, and Stochastic Gradient Descent.

| Solver | Learned θ / computation | Iterations or epochs |
|---|---|---|
| Normal Equation | Direct closed-form solution | No iterations |
| Batch Gradient Descent | Iterative gradient-based solution | 20,000 iterations for the final model |
| Stochastic Gradient Descent | Iterative sample-based solution | 50 epochs for the final SGD configuration |

For the supplied Set A results, Batch Gradient Descent produced values extremely close to the Normal Equation:

| Feature | Normal Equation | Batch GD | Absolute difference |
|---|---:|---:|---:|
| Intercept | 6803.188733 | 6803.188734 | 0.000000 |
| Irradiation | 8187.254344 | 8184.953498 | 2.300846 |
| Module temperature | 5.329879 | 8.425158 | 3.095278 |
| Ambient temperature | -65.130812 | -66.092969 | 0.962156 |
| sin_hour | -86.115151 | -86.150016 | 0.034864 |
| cos_hour | -606.321145 | -606.290027 | 0.031117 |

These results show that Batch Gradient Descent reached a solution very close to the Normal Equation solution.

SGD produced noticeably different coefficients, especially for module temperature:

| Feature | Normal Equation | SGD |
|---|---:|---:|
| Intercept | 6803.188733 | 6811.211897 |
| Irradiation | 8187.254344 | 7597.120180 |
| Module temperature | 5.329879 | 832.669598 |
| Ambient temperature | -65.130812 | -257.373670 |
| sin_hour | -86.115151 | -76.944338 |
| cos_hour | -606.321145 | -563.018271 |

Therefore, the three solvers did **not** reach exactly the same θ values. Batch GD is much closer to the Normal Equation solution than the supplied SGD result.

For this dataset, the **Normal Equation** is a strong choice because the number of features is small and the dataset is manageable. It gives a direct solution without requiring learning-rate tuning or many iterations. For a dataset with **10 million rows**, an iterative method such as **SGD** would generally be more suitable because it can process individual samples or mini-batches and is more scalable to very large datasets. The final choice should be consistent with the Optimization Decision Matrix from Lecture 1.

## 5.4 Batch GD vs SGD J(θ) Curves

The Batch Gradient Descent and Stochastic Gradient Descent cost curves should be compared using the generated learning-curve figures.

The **Batch GD curve is expected to be smoother** because each parameter update uses the gradient calculated from the complete training set. The update therefore represents the overall direction of the cost function more consistently.

The **SGD curve is noisier** because each update is based on an individual training example (or a stochastic sample). Different samples produce different gradient estimates, causing the cost to fluctuate rather than decrease smoothly at every update.

Therefore:

- Batch GD → smoother and more stable J(θ) curve.
- SGD → noisier/fluctuating J(θ) curve, but potentially more scalable for very large datasets.

The generated figure should be included with the final report/blog as evidence for this comparison.

## 5.5 Residuals vs Hour of Day — Set A Normal Equation

For the Set A Normal Equation model, the residual is defined as:

\[
r = y - h_\theta(x)
\]

where \(y\) is the measured AC power and \(h_\theta(x)\) is the model prediction.

The residuals should be plotted against hour of day using the generated residual plot.

The supplied θ table alone does **not** contain the residual-by-hour values, so the exact worst hours cannot be determined from the θ table alone. The exact hours should be identified from `results/figures/6_residuals_vs_hour.png` or the residual statistics generated by the program.

In interpreting the plot, hours with the largest positive or negative residual magnitudes represent periods where the model performs worst. Large errors around sunrise or sunset can be physically plausible because solar irradiation changes rapidly during these periods. Changing sun angle, partial shading, atmospheric conditions, and nonlinear PV/inverter behavior can also contribute to larger prediction errors.

### Final Task 5.5 statement to complete after running the residual analysis

> The largest residual errors occurred around (figures/6_residuals_vs_hour.png) in results. These errors can be explained by rapid changes in solar irradiation and solar geometry, particularly near sunrise and sunset. Other possible physical effects include temporary shading, atmospheric conditions, and nonlinear PV/inverter behavior that are not fully captured by the linear model.

