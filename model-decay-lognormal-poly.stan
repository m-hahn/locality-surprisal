data {
   int<lower=1> N; // number of datapoints
   real time[N]; // readig time
   int M; // items
   int L; // participants
   int<lower=1,upper=M> tokenID[N];
   int<lower=1,upper=L> Participant[N];
   vector<lower=-100,upper=100>[N] LogWordFreq;
   matrix<lower=-100,upper=100>[N,35] Increments;
   //vector<lower=-100,upper=100>[N] ExperimentTokenLength;
}
parameters {
  real alpha; // intercept
  matrix[1,M]  z_tokenID;
  matrix[3,L]  z_Participant;
  real<lower=0> sigma_e;
  vector<lower=0>[1] sigmaSlope_tokenID;
  vector<lower=0>[3] sigmaSlope_Participant;
  cholesky_factor_corr[1] L_tokenID;
  cholesky_factor_corr[3] L_Participant;
  real beta1;
  real lambda;
  real kappa;
}
transformed parameters{
   matrix[1,M] for_tokenID;
   matrix[3,L] for_Participant;
   for_tokenID = diag_pre_multiply(sigmaSlope_tokenID,L_tokenID) * z_tokenID;
   for_Participant = diag_pre_multiply(sigmaSlope_Participant,L_Participant) * z_Participant;
}
model {
  vector[35] distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]';
  vector[35] contributions = exp(lambda * log(distances));

  vector[N] withContributions = Increments * contributions;

  L_tokenID ~ lkj_corr_cholesky(2.0);
  L_Participant ~ lkj_corr_cholesky(2.0);
  to_vector(z_tokenID) ~ normal(0,1);
  to_vector(z_Participant) ~ normal(0,1);
  
 for (n in 1:N) {
   int tokenIDForN;
   int ParticipantForN;
   real gamma_mean;
   real alpha_par;
   tokenIDForN = tokenID[n];
   ParticipantForN = Participant[n];
   
   gamma_mean = alpha +for_tokenID[1,tokenIDForN] + for_Participant[1,ParticipantForN] + LogWordFreq[n] * (beta1 + for_Participant[2,ParticipantForN]) + withContributions[n] * (kappa + for_Participant[3, ParticipantForN]);
   time[n] ~ normal( gamma_mean , sigma_e);
   //alpha_par = fmax(10.0,gamma_mean) * sigma_e;
   //time[n] ~ gamma(alpha_par, sigma_e);
 }
}

