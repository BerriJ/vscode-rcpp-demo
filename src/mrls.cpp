#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

//' @export
// [[Rcpp::export]]
arma::mat mrls(const arma::vec &y,
               const arma::mat &x,
               const int &start,
               const arma::mat &rhoseq,
               const arma::vec &refitseqby,
               const double &eps,
               const double &lambda)
{
    int nT = y.n_elem;
    int nTX = x.n_rows;
    int k = x.n_cols;
    int nrho = rhoseq.n_cols;
    int jdx = 0;

    vec Kmtx(k, fill::zeros);
    mat kk(1, 1);
    cube beta(nT - start + 1, k, nrho, fill::zeros);
    cube betao(nT - start + 1, k, nrho, fill::zeros);
    mat fit(nT - start + 1, nrho);
    vec wk(k);
    mat xpxi(k, k);
    mat diagmatlambda(k, k, fill::eye);
    diagmatlambda *= lambda;

    for (int irho = 0; irho < nrho; irho++)
    { // nrho

        double rho = rhoseq(nT - 2, irho);
        int cutn = (int)(log(eps) / log(rho)) + 1;

        vec refitseq = regspace(start, refitseqby(irho), ((double)nT) + refitseqby(irho));

        int h = 0;
        vec w = linspace<vec>(0, 1, nT - .5) * 0 + 1;
        for (int t = start; t < nT; t++)
        { // nT
            jdx = t - start;
            if (t == refitseq(h))
            {
                int axt = (t + cutn - abs(t - cutn)) / 2;
                w = (rhoseq.col(irho)).tail(axt);
                mat xr = (x.rows(t - axt, t - 1));
                colvec yr = y.subvec(t - axt, t - 1);
                colvec wts = sqrt(w);
                xpxi = (((xr.each_col() % wts).t() * (xr.each_col() % wts)) + diagmatlambda * sum(w)).i(); // diagmatlambda
                betao(span(jdx), span(0, k - 1), span(irho)) = beta(span(jdx), span(0, k - 1), span(irho));
                beta(span(jdx), span(0, k - 1), span(irho)) = (xpxi * (((xr.each_col() % wts).t()) * (yr % wts))).t();
                h = h + 1;
            }
            wk = (x.row(t) * xpxi).t();
            kk = (1 + wk.t() * (x.row(t)).t());
            xpxi = (xpxi - (wk * wk.t()) / as_scalar(rho + wk.t() * (x.row(t)).t())) / rho;
            Kmtx = xpxi * (x.row(t)).t();
            rowvec tmpbeta = beta(span(jdx), span(0, k - 1), span(irho));

            beta(span(jdx + 1), span(0, k - 1), span(irho)) = tmpbeta - (Kmtx.t() * as_scalar(x.row(t) * trans(tmpbeta) - y(t)));
        } // t
        fit.col(irho) = sum((x.rows(nTX - (nT - start + 1), nTX - 1)) % beta.slice(irho), 1);
    } // rho
    return fit;
}