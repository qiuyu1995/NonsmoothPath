#' Path-following algorithm for L2 penalized nonsmooth problems
#'
#' @description
#' Compute exact solution path for case-weight adjusted problem
#' \loadmathjax
#' \mjseqn{(\beta_{0,w}, \beta_{w}) = argmin_{\beta_0, \beta} \sum_{i \neq j} f(g_i(\beta_0, \beta)) + w*f(g_{j}(\beta_0, \beta)) + \lambda / 2 * \|\beta\|_2^2}
#' for \mjseqn{0 <= w <= 1}, where \mjseqn{g_i(\beta_0, \beta) = a_i \beta_0 + b_i^T \beta + c_i} and \mjseqn{f(r) = \alpha_0 \max(r, 0) + \alpha_1 \max(-r, 0).}
#' The exact solution path is shown to be piece-wise linear
#' and the function will return all the breakpoints and their solutions.
#'
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param obs_index Index of the observation that is attached a weight, ranging from 1 to n
#' @param class Specify the problem class with default value "quantile". Use "quantile" for quantile regression, and "svm" for support vector machine
#' @param tau The parameter for quantile regression, ranging between 0 and 1. No need to specify the value of it if choose class = "svm"
#'
#' @return W_vec A list of breakout points
#' @return Beta_0 True values of beta_{0,w} at breakout points
#' @return Theta True values of theta_{0,w} at breakout points
#' @return Beta True values of beta_{w} at breakout points
#' @export

nonsmooth_path <- function(X, Y, lam, obs_index, class = "quantile", tau = 0.5){
  n = dim(X)[1]
  p = dim(X)[2]
  if (class == "quantile") {
    a = -rep(1, n)
    B = -X
    c = Y
    alpha_0 = tau
    alpha_1 = 1 - tau
  }
  else if (class == "svm") {
    a = Y
    B = matrix(0, nrow = n, ncol = p)
    for (i in 1:n)
      B[i, ] = X[i, ] * Y[i]
    c = -rep(1, n)
    alpha_0 = 0
    alpha_1 = 1
  }
  else {
    cat("\n The current version only supports 'quantile' and 'svm' class options.\n")
    return(0)
  }

  ## Compute the full-data solution
  beta <- CVXR::Variable(p)
  beta_0 = CVXR::Variable()
  xi = CVXR::Variable(n)
  eta = CVXR::Variable(n)
  obj = alpha_0 * CVXR::sum_entries(xi) + alpha_1 * CVXR::sum_entries(eta) + lam / 2 * CVXR::power(CVXR::norm2(beta), 2)
  constraints = list(xi >= 0, eta >= 0, beta_0 * a + B %*% beta + c + eta >= 0, - beta_0 * a - B %*% beta - c + xi >= 0)
  prob <- CVXR::Problem(Minimize(obj), constraints)
  result <- CVXR::solve(prob, reltol=1e-12)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  ## Run the path-following algorithm
  .Call(`_NonsmoothPath_case_path_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, obs_index-1, beta_0_w0, beta_w0, theta_w0)
}