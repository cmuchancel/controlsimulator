# Next Steps

1. Add richer plant encodings.
   Include pole-zero summaries, Bode samples, and time-normalized features instead of relying mostly on padded coefficients.

2. Improve the regressor around oscillatory tails.
   The current MLP is strong on trajectory RMSE but still underperforms on settling-time coverage for hard oscillatory cases.

3. Sample more densely near the stability boundary.
   Active or curriculum-style gain sampling would make the classifier harder in useful ways and should improve boundary sharpness.

4. Extend the simulator scope.
   Add delays, actuator saturation, and measurement noise once the core LTI pipeline is stable.

5. Compare against one stronger architecture.
   A small decoder head or residual MLP is the next reasonable comparison before considering more exotic sequence models.
