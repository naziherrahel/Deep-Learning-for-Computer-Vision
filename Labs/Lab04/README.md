# Lab 04 — Linear Scores and Multiclass SVM Loss 



## Learning goals

- compute class scores with a linear model;
- implement the multiclass SVM hinge loss using explicit loops;
- derive the analytic gradient with respect to the weight matrix;
- verify the gradient with centered finite differences;
- explain how the margin affects the loss and gradient.

## Files

- `lab04_svm_loss.ipynb` — guided exercises and public checks;
- `linear_classifier.py` —  implementation of the naive SVM loss;

## Suggested timing

1. Linear-score warm-up — 10 minutes
2. Hinge-loss implementation — 25 minutes
3. Analytic gradient — 30 minutes
4. Numerical gradient checks — 15 minutes
5. Interpretation and wrap-up — 10 minutes

Work directly in the local files. No external platform integration or upload
step is included.

## Completion checklist

- The unregularized fixture loss is `2.0`.
- Sparse gradient-check relative errors are normally below `1e-5`.
- Every Lab 04 call uses `reg=0.0`; L2 regularization begins in Lab 05.
- `linear_classifier.py` contains no remaining `NotImplementedError` in
  `svm_loss_naive`.
