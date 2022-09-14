// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

/*  Function dealing with the case when E is empty

 When E is empty, beta_0 is not unique and is between [Low, High]. We set beta_0 to be either Low or High depend on j is in L or R so that E becomes non-empty.
*/
arma::vec case_weight_adjusted_Empty_E(arma::vec a, arma::mat B, arma::vec c, arma::vec beta_w, arma::uvec L, arma::uvec R, int j_in_L, int j) {
	
	double beta_0_temp = 0.0;
	double beta_0_update = 0.0;
	int index_move_to_E = 0;
	arma::vec output(2); 
	int L_size = L.size();
	int R_size = R.size();

/*	if (j_in_L) {
		// Set beta_0 as High
		beta_0_temp = (- B.rows(R) * beta_w - c.elem(R)) / a.elem(R);
		beta_0_update = beta_0_temp.min();
		index_move_to_E = (int) beta_0_temp.index_min();   
	} else {
		// Set beta_0 as Low
		beta_0_temp = (- B.rows(L) * beta_w - c.elem(L)) / a.elem(L);
		beta_0_update = beta_0_temp.max();
		index_move_to_E = - (int) beta_0_temp.index_max() - 1;
	}*/

	if ((j_in_L && a[j] < 0) || (!j_in_L && a[j] > 0)) {
		beta_0_update = 1e8;
		// Set beta_0 as High
		for (unsigned i=0; i<L_size; i++) {
			if (a[L[i]] > 0) {
				beta_0_temp = (- arma::dot(B.row(L[i]), arma::trans(beta_w)) - c[L[i]]) / a[L[i]];
				if (beta_0_temp < beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = - (int) i - 1;
				}
			}
		}
		for (unsigned i=0; i<R_size; i++) {
			if (a[R[i]] < 0) {
				beta_0_temp = (- arma::dot(B.row(R[i]), arma::trans(beta_w)) - c[R[i]]) / a[R[i]];
				if (beta_0_temp < beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = (int) i;
				}
			}
		}
	} else {
		beta_0_update = -1e8;
		// Set beta_0 as Low
		for (unsigned i=0; i<L_size; i++) {
			if (a[L[i]] < 0) {
				beta_0_temp = (- arma::dot(B.row(L[i]), arma::trans(beta_w)) - c[L[i]]) / a[L[i]];
				if (beta_0_temp > beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = - (int) i - 1;
				}
			}
		}
		for (unsigned i=0; i<R_size; i++) {
			if (a[R[i]] > 0) {
				beta_0_temp = (- arma::dot(B.row(R[i]), arma::trans(beta_w)) - c[R[i]]) / a[R[i]];
				if (beta_0_temp > beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = (int) i;
				}
			}
		}
	}
	output[0] = beta_0_update;
	output[1] = index_move_to_E;
	return output;
}


// Update or downdate the inverse matrix by changing only one point
arma::mat case_weight_adjusted_update(arma::mat K_inv, arma::vec a, arma::mat B, arma::uvec E, int E_size) {
       
	arma::uword k = E_size;
	arma::uword index_insert = E[k-1];
	arma::mat K_inv_new(k+1, k+1);
	arma::uvec E_prev = E.subvec(0, k-2);
	arma::mat B_E_prev = B.rows(E_prev);
	
	arma::vec v = (B.row(index_insert)).t();
	arma::vec u1(k, arma::fill::ones);
	u1 *= a(index_insert);
    u1.tail(k-1) = B_E_prev * v;
	arma::vec u2 = K_inv* u1;	
	double d = 1.0/(arma::dot(v, v) - dot(u1, u2));	
	arma::vec u3 = d * u2;
	arma::mat F11_inv = K_inv + d * u2 * u2.t();
	K_inv_new(k, k) = d;
	K_inv_new(arma::span(0, k-1), k) = -u3;
	K_inv_new(k, arma::span(0, k-1)) = -u3.t();
	K_inv_new(arma::span(0, k-1), arma::span(0, k-1)) = F11_inv;

	return K_inv_new;
}

arma::mat case_weight_adjusted_downdate(arma::mat K_inv, arma::mat B, arma::uvec E_prev, int E_prev_size, int index_remove) {

	arma::uword k = E_prev_size;
	arma::mat K_inv_new(k, k);
	
	if (index_remove < k-1) {
		arma::rowvec tmpv1 = K_inv.row(index_remove+1);
		K_inv.rows(index_remove+1, k-1) = K_inv.rows(index_remove+2, k);
		K_inv.row(k) = tmpv1;
		arma::vec tmpv2 = K_inv.col(index_remove+1);
		K_inv.cols(index_remove+1, k-1) = K_inv.cols(index_remove+2, k);
		K_inv.col(k) = tmpv2;
	}
	arma::mat F11_inv = K_inv.submat(0, 0, k-1, k-1);
	double d = K_inv(k, k);
	arma::vec u = - K_inv(arma::span(0, k-1), k) / d;
	K_inv_new = F11_inv - d * u * u.t();
	return K_inv_new;
}

arma::vec case_weight_adjusted_Compute_b(arma::vec a, arma::mat B, arma::uvec E, int E_size, int j, double T, arma::mat K_inv) {
	arma::vec y_et(E_size + 1, arma::fill::zeros);
	y_et(0) = -a(j) * T;
    y_et.tail(E_size) = -B.rows(E) * B.row(j).t() * T;
    arma::vec b = K_inv * y_et;
    return b;
}

arma::mat case_weight_adjusted_Compute_K_inv(arma::vec a, arma::mat B, arma::uvec E, int E_size) {
	arma::mat B_elb = B.rows(E);
    arma::mat K(E_size + 1, E_size + 1, arma::fill::zeros);
    K(arma::span(1, E_size), 0) = a.elem(E) ;
    K(0, arma::span(1, E_size)) = a.elem(E).t();
    K(1,1,arma::size(E_size,E_size)) = B_elb * B_elb.t();
    arma::mat K_inv = arma::inv(K);
    return K_inv;
}

//'Case-weight adjusted solution path for L2 regularized nonsmooth problem (quantile regression and svm)
//'
//' @description
//' Path-following algorithm to exactly solve 
//' 	(beta_{0,w}, beta_{w}) = argmin_{beta_0, beta} \sum_{i \neq j} f(g_i(beta_0, beta)) + w*f(g_{j}(beta_0, beta)) + lambda / 2 * \|beta\|_2^2
//' for 0 <= w <= 1, where g_i(beta_0, beta) = a_i beta_0 + b_i^T beta + c_i and f(r) = alpha_0 max(r, 0) + alpha_1 max(-r, 0)
//'
//' @param a A \eqn{n \times 1} vector and a^T = (a_1, ..., a_n)
//' @param B A \eqn{n \times p} matrix and B^T = (b_1, ..., b_n)
//' @param c A \eqn{n \times 1} matrix and c^T = (b_1, ..., b_n) 
//' @param lam Regularization parameter for L2 penalty
//' @param alpha_0 A scalar in the definition of f(r)
//' @param alpha_1 A scalar in the definition of f(r)
//' @param j Index of the observation that is attached a weight
//' @param beta_0_w0 A scalar, which is the true value of beta_{0,w} when w = w_0 = 1
//' @param beta_w0 A \eqn{p \times 1} vector, which is the true value of beta_{w} when w = w_0 = 1
//' @param theta_w0 A \eqn{n \times 1} vector, which is the true value of the dual variable when w = w_0 = 1
//'
//' @details
//' This function will be called by function CaseInfluence_nonsmooth to generate case influence graph for each case.
//'
//' @return W_vec A list of breakout points
//' @return Beta_0 True values of beta_{0,w} at breakout points
//' @return Theta True values of theta_{w} at breakout points
//' @return Beta True values of beta_{w} at breakout points
// [[Rcpp::export(case_path_nonsmooth)]]
Rcpp::List case_path_nonsmooth(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, int j, 
	double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0){
	const int n = B.n_rows;
	const int p = B.n_cols;

	const int N = 1000;
	
  	// Store breakpoints and solutions
	arma::vec Beta_0(N+1);
	Beta_0(0) = beta_0_w0;
	arma::vec W_vec(N+1);
	W_vec(0) = 1; 
	arma::mat Theta;
	Theta.insert_cols(Theta.n_cols, theta_w0);
	arma::vec r = beta_0_w0 * a + B * beta_w0 + c;
	arma::mat R_mat;
	R_mat.insert_cols(R_mat.n_cols, r);
	arma::mat Beta;
	Beta.insert_cols(Beta.n_cols, beta_w0);
	double beta_0_w = beta_0_w0;
	arma::vec theta_w = theta_w0;
	arma::vec beta_w = beta_w0;
	
	// Declare and initialize three elbow sets, and theta
	arma::uvec E;
	arma::uvec L;
	arma::uvec R;
	int E_size = 0;
	int L_size = 0;
	int R_size = 0;

	arma::uvec index_insert(1);
	int index_j = -1;
	int j_in_L = -1;
	arma::vec theta_insert(1);

    // Initialize E/L/R and theta_E
	const double epsilon_theta = 1e-8;
	const double epsilon_r = 1e-5;
	for (unsigned i=0; i<n; i++){
		index_insert(0) = i;
		if (fabs(theta_w0[i]+alpha_0) < epsilon_theta || r[i] > epsilon_r){
			R.insert_rows(R_size, index_insert);
			R_size = R_size + 1;
			if (i == (unsigned) j){
				j_in_L = 0;
			}
		} else if (fabs(theta_w0[i]-alpha_1) < epsilon_theta || r[i] < -epsilon_r){
			L.insert_rows(L_size, index_insert);
			L_size = L_size + 1;
			if (i == (unsigned) j){
				j_in_L = 1;
			}
		} else {
			E.insert_rows(E_size, index_insert);
			if (i == (unsigned) j) {
			  index_j = E_size;  // The index of j in set E
			}
			E_size = E_size + 1;
		}
	}

    // Declare variables 
	int m = 0;
	double w_m = 1.0;
	double w_m_next = 1.0;

	double T = 0.0;
	arma::vec b_m;
	arma::vec h_m(n);
	arma::mat K_inv;

	arma::vec w_1_alpha1_temp;
	double w_1_alpha1_max = 0.0;
	int w_1_alpha1_index = 0;
	arma::vec w_1_alpha0_temp;
	double w_1_alpha0_max = 0.0;
	int w_1_alpha0_index = 0;
	double w_1_max = 0.0;
	int w_1_index = 0;

	arma::vec w_2_L_temp;
	double w_2_L_max = 0.0;
	int w_2_L_index = 0;
	arma::vec w_2_R_temp;
	double w_2_R_max = 0.0;
	int w_2_R_index = 0;
	double w_2_max = 0.0;
	int w_2_index = 0;
	int w_2_L_is_max = 1;
	const double INF = 1e8;

	int index_in_elbow = 0;
	arma::vec E_empty_output(2);
	double beta_0_update = 0.0;

	// Initialize K_inv
	if (E_size) {
		K_inv = case_weight_adjusted_Compute_K_inv(a, B, E, E_size);
	}

	// Manage the case when j is in E
	if (j_in_L == -1) {
		// Find the next breakpoint
		if (theta_w(j) > 0){
			w_m = theta_w(j) / alpha_1;
			j_in_L = 1;
		} else {
			w_m = theta_w(j) / (-alpha_0);
			j_in_L = 0;
		}

		// Update three sets and theta_E
		if (w_m > 0){
			index_insert(0) = j;
			if (j_in_L){
				L.insert_rows(L_size, index_insert);
				L_size = L_size + 1;
			} else {
				R.insert_rows(R_size, index_insert);
				R_size = R_size + 1;
			}
			if (E_size > 1){
				K_inv = case_weight_adjusted_downdate(K_inv, B, E, E_size, index_j);
            }
			E.shed_row(index_j);
			E_size = E_size - 1;
			r(j) = 0;
		}
		m = m + 1;
		Beta_0(m) = beta_0_w0;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
		Beta.insert_cols(Beta.n_cols, beta_w);
		W_vec(m) = w_m; 
	}

	// Case when alpha_0 = 0 and j is in R (Algorithm terminates)
	if (alpha_0 == 0 && j_in_L == 0){
		m = m + 1;
		Beta_0(m) = Beta_0(m-1);
		w_m = 0;
		W_vec(m) = w_m;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
		Beta.insert_cols(Beta.n_cols, beta_w);
	}

	// Compute T (It will not change)
	if (j_in_L) {
		T = alpha_1;
	} else {
		T = -alpha_0;
	}

	while (w_m > 1e-6) {
		//std::cout << "m is " << m << std::endl;
		//std::cout << "w_m is " << w_m << std::endl;
		//std::cout << "starting: E is " << E << std::endl;
		//std::cout << "K_inv is " << K_inv << std::endl;
		if (E_size > 0) {
			// Compute b = (b_0, b_1, ..., b_{|E|}) and h_m
			b_m = case_weight_adjusted_Compute_b(a, B, E, E_size, j, T, K_inv);
			//std::cout << "b_m is " << b_m << std::endl;
			h_m = a * b_m(0) + B * (arma::trans(B.rows(E)) * b_m.tail(E_size) + B.row(j).t() * T);

			// Compute candidate w_1m
			// w_1_index is the case in E that should be moved to L/R, and w_1_max is the candidate w_1m
			w_1_max = -INF;
			if (E_size > 0) {
				w_1_alpha1_max = -INF;
				w_1_alpha1_temp = (- theta_w.elem(E) + alpha_1) / b_m.tail(E_size) + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha1_temp(i) < w_m) && (w_1_alpha1_temp(i) > w_1_alpha1_max)) {
						w_1_alpha1_max = w_1_alpha1_temp(i);
						w_1_alpha1_index = i;
					}
				}

				w_1_alpha0_max = -INF;
				w_1_alpha0_temp = (- theta_w.elem(E) - alpha_0) / b_m.tail(E_size) + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha0_temp(i) < w_m) && (w_1_alpha0_temp(i) > w_1_alpha0_max)) {
						w_1_alpha0_max = w_1_alpha0_temp(i);
						w_1_alpha0_index = i;
					}
				}

				if (w_1_alpha0_max > w_1_alpha1_max) {
					w_1_max = w_1_alpha0_max;
					w_1_index = w_1_alpha0_index;
				} else {
					w_1_max = w_1_alpha1_max;
					w_1_index = w_1_alpha1_index;
				}
			}
			//std::cout << "w_1_max is " << w_1_max << std::endl;

			// Compute candidate w_2m
			// w_2_index is the case in L/R that should be moved to E, and w_2_L_is_max indicates this case is from L or R
			// w_2_max is the candidate w_2m
			w_2_max = -INF;
			w_2_L_max = -INF;
			w_2_R_max = -INF;
			if (L_size > 0) {
				w_2_L_temp = - lam * r.elem(L) / h_m.elem(L) + w_m;
				for (unsigned int i=0; i<L_size; i++) {
					if ((w_2_L_temp(i) < w_m) && (w_2_L_temp(i) > w_2_L_max)) {
						w_2_L_max = w_2_L_temp(i);
						w_2_L_index = i;
					}
				}
			}

			if (R_size > 0) {
				w_2_R_temp = - lam * r.elem(R) / h_m.elem(R) + w_m;
				for (unsigned int i=0; i<R_size; i++) {
					if ((w_2_R_temp(i) < w_m) && (w_2_R_temp(i) > w_2_R_max)) {
						w_2_R_max = w_2_R_temp(i);
						w_2_R_index = i;
					}
				}
			}

			if (w_2_L_max > w_2_R_max) {
				w_2_max = w_2_L_max;
				w_2_index = w_2_L_index;
				w_2_L_is_max = 1;
			} else {
				w_2_max = w_2_R_max;
				w_2_index = w_2_R_index;
				w_2_L_is_max = 0;
			}
			//std::cout << "w_2_max is " << w_2_max << std::endl;
			w_m_next = std::max(w_1_max, w_2_max);
			
			// Compute beta, r at the next breakpoint
			beta_0_w += b_m(0) / lam * (w_m_next - w_m);
			theta_w.elem(E) += b_m.tail(E_size) * (w_m_next - w_m);
			theta_w(j) = w_m_next * T;
			r += h_m / lam * (w_m_next - w_m);
			beta_w += (arma::trans(B.rows(E))*b_m.tail(E_size) + T*B.row(j).t()) * (w_m_next - w_m) / lam;

			// Update three elbow sets and theta_E
			if (w_m_next == w_1_max) {
				// The case when E moves an element to either L or R
				index_insert(0) = E[w_1_index];				
				if (fabs(theta_w[E[w_1_index]] - alpha_1) < epsilon_theta) {
					L.insert_rows(L_size, index_insert);
					L_size = L_size + 1;
				} else {
					R.insert_rows(R_size, index_insert);
					R_size = R_size + 1;
				}
				if (E_size > 1){
					K_inv = case_weight_adjusted_downdate(K_inv, B, E, E_size, w_1_index);
                }
				r(E[w_1_index]) = 0;
				E.shed_row(w_1_index);
				E_size = E_size - 1;
			} else {
				// The case when L/R moves an element to E
				if (w_2_L_is_max) {
					index_insert(0) = L[w_2_index];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_w[L[w_2_index]] = alpha_1;
					E_size = E_size + 1;  
					L.shed_row(w_2_index);
					L_size = L_size - 1;
				} else {
					index_insert(0) = R[w_2_index];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_w[R[w_2_index]] = -alpha_0;
					E_size = E_size + 1;  
					R.shed_row(w_2_index);
					R_size = R_size - 1;					
				}
				K_inv = case_weight_adjusted_update(K_inv, a, B, E, E_size);
			}
		} else {
			//std::cout<< "Deal with the empty E case" << std::endl;
			// The case when E is empty
		    E_empty_output = case_weight_adjusted_Empty_E(a, B, c, beta_w, L, R, j_in_L, j);
		    beta_0_update = E_empty_output[0];
		    r = r + (beta_0_update - beta_0_w) * a;   // Update the residual
		    beta_0_w = beta_0_update;
		    index_in_elbow = E_empty_output[1];
		    // Move an element from L/R to E
		    if (index_in_elbow < 0) {
					index_insert(0) = L[-index_in_elbow-1];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_w[L[-index_in_elbow-1]] = alpha_1;
					E_size = E_size + 1;  
					L.shed_row(-index_in_elbow-1);
					L_size = L_size - 1;		
				} else {
					index_insert(0) = R[index_in_elbow];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_w[R[index_in_elbow]] = -alpha_0;
					E_size = E_size + 1;  
					R.shed_row(index_in_elbow);
					R_size = R_size - 1;
		    }
			Beta_0(m) = beta_0_w;
			R_mat.col(m) = r;

			K_inv = case_weight_adjusted_Compute_K_inv(a, B, E, E_size);			
			continue;
		}	
		m = m + 1;
		Beta_0(m) = beta_0_w;
		W_vec(m) = w_m_next; 
		w_m = w_m_next;
		Theta.insert_cols(Theta.n_cols, theta_w);
		R_mat.insert_cols(R_mat.n_cols, r);
		Beta.insert_cols(Beta.n_cols, beta_w);
		//std::cout << "ending: E is " << E << std::endl;
	}

	arma::vec Beta_0_output(m+1);
	Beta_0_output = Beta_0.subvec(0, m);

	arma::vec W_vec_output(m+1);
	W_vec_output = W_vec.subvec(0, m);

	return Rcpp::List::create(Rcpp::Named("W_vec") = W_vec_output,
	                          Rcpp::Named("Beta_0") = Beta_0_output,
	                          Rcpp::Named("Theta") = Theta,
	                          Rcpp::Named("Beta") = Beta); 
}