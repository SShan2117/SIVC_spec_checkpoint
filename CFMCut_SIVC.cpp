// Charge-neutral TBG. All units in international units unless specified.
#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_scalapack.h"
#include "mpi.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using std::cout;     // output to the control panel
using std::endl;     // change line
using std::ifstream; // red file
using std::ios;      // input(ios::in) and output(ios::out)
using std::ofstream; // ofstream substitutes std::ofstream; create and write file
double pi = 3.141592653589793, e = 1.602176487e-19, eps0 = 8.854187817e-12, hbar = 1.054571628e-34;
double G0[2][2] = {{-sqrt(3.0) / 2.0, -1.5}, {sqrt(3.0), 0.0}},                                                      // Bases of mBZ with module |G|
    K[2][2] = {{-sqrt(3.0) / 2.0, 0.5}, {-sqrt(3.0) / 2.0, -0.5}},                                                   // Dirac points in mBZ
    G1[4][2] = {{sqrt(3.0) / 2.0, 1.5}, {-sqrt(3.0) / 2.0, -1.5}, {-sqrt(3.0) / 2.0, 1.5}, {sqrt(3.0) / 2.0, -1.5}}, // Vectors of four directions
    BMcut = 108.1;                                                                                                   // Momentum cutoff of BM model, six times of |G|

const int _nG = 8,  // Eight times of |G| to cover BMcut
    nG = 127,       // Total number of momentum points for BMcut
    ncb = 2,        // Two flat bands
    nk0 = 9,        // Linear size of momentum smearing in mBZ

    ntau = 150,     // Number of imaginary time slices
    Nmc = 3000,      // Number of mc steps
    Smc = 300,      // Thermolization steps

    ifcontinue = 1, // If continue from last run
    nk = nk0 * nk0, // Momentum smearing in mBZ, system size
    lnG = nG * 4,   // Linear size of BM model matrix
    LnG = lnG * lnG,
          lnk = nk * ncb, // Linear size of operator matrix
    Lnk = lnk * lnk;
double u0, // AA inter-layer hopping
    u1,    // AB inter-layer hopping
    dgate,
    Istrength,
    theta = 1.08 / 180.0 * pi, // Twisting angle
    Tstart,                    // Starting simulation temperature in meV
    Tend = 0.01,               // Ending T
    Tstep = 50.1,
    nu = 0.0,                                                                // Filling
    eps = 7.0 * eps0,                                                        // Dielectric
    a = 1.42e-10,                                                            // Nearest distance of carbon atoms
    hvF = 2.37745 * e * sqrt(3.0) * a,                                       // Fermi velocity for flat bands
    rp_unit = 8.0 * pi * sin(theta / 2.0) / (3.0 * sqrt(3.0) * a),           // Units for momentum space
    Omega = 3.0 * sqrt(3.0) * nk * a * a * 0.125 / pow(sin(theta * 0.5), 2), // Real space area
    G[nG * 2],                                                               // Array to store momentum points for BM model
    S[(ntau + 1) * lnk],                                                     // Array to store singlar values of matrices
    Stmp[(ntau + 1) * lnk],                                                  // store S before unequal-time update
    superb[lnk - 1],                                                         // Working space of SVD
    DR[2 * lnk],                                                             // Array for calculating Green's function
    DL[2 * lnk];

MKL_Complex16 alpha = {1.0, 0.0},
              beta = {0.0, 0.0},
              c0 = {0.0, 0.0};
int direc[2] = {-1, 1}; // For update directions

MKL_Complex16 RM[Lnk], RN[Lnk], C[Lnk]; // Arrays frequently used
MKL_INT info, Ipiv[lnk];                // Working area for LAPACKE

double square(double x) { // Square
    return x * x;
}

double abs1(double x) { // Absolute value
    if (x < 0.0) {
        return -x;
    } else {
        return x;
    }
}

void BM_eigen(double *k, double *w, MKL_Complex16 *vr, int ifbands) { // input k, output w and vr;ifband=0 for form factor,ifband=1 for 20 bands
    MKL_Complex16 *Hkk = new MKL_Complex16[LnG];                      // BM model matrix
    int Ia, check[lnG], Bool;
    for (int i = 0; i < LnG; i++)
        Hkk[i] = c0;
    double _k[nG * 2], val[3], dw[lnG];
    for (int i = 0; i < nG; i++)
        for (int l = 0; l < 2; l++)
            _k[i * 2 + l] = k[l] + G[i * 2 + l]; // The k point and its extented BZ counterparts
    for (int i = 0; i < nG; i++) {               // i stands for the number of extended G
        for (int i1 = 0; i1 < 2; i1++) {         // i1 stands for layers
            for (int l = 0; l < 2; l++)          // l stands for real or imaginary
                val[l] = -hvF * (_k[i * 2 + l] - K[i1][l]) * rp_unit;
            for (int i2 = 0; i2 < 2; i2++) { // i2 stands for sublattices
                Ia = i * nG * 16 + i1 * nG * 8 + i2 * nG * 4 + i * 4 + i1 * 2 + 1 - i2;
                Hkk[Ia].real = val[0];
                Hkk[Ia].imag = -val[1] * pow(-1.0, i2); // Intralayer hopping
                Ia = i * nG * 16 + i1 * nG * 8 + i2 * nG * 4 + i * 4 + (1 - i1) * 2 + i2;
                Hkk[Ia].real = u0; // Interlayer hopping for same k point
                Ia = i * nG * 16 + i1 * nG * 8 + i2 * nG * 4 + i * 4 + (1 - i1) * 2 + 1 - i2;
                Hkk[Ia].real = u1;
            }
        }
        for (int j = 0; j < nG; j++)
            if (j != i)
                for (int i0 = 0; i0 < 4; i0++)
                    if (abs1(_k[i * 2] - _k[j * 2] - G1[i0][0]) < 1.0e-8 && abs1(_k[i * 2 + 1] - _k[j * 2 + 1] - G1[i0][1]) < 1.0e-8) // Check for four directions, interlayer hopping for different k points
                        for (int i2 = 0; i2 < 2; i2++) {
                            Ia = i * nG * 16 + (i0 % 2) * nG * 8 + i2 * nG * 4 + j * 4 + (1 - i0 % 2) * 2 + i2;
                            Hkk[Ia].real = u0;
                            Ia = i * nG * 16 + (i0 % 2) * nG * 8 + i2 * nG * 4 + j * 4 + (1 - i0 % 2) * 2 + 1 - i2;
                            val[0] = pow(-1.0, i2 + 1 + i0 / 2) * 2.0 * pi / 3.0;
                            Hkk[Ia].real = u1 * cos(val[0]);
                            Hkk[Ia].imag = u1 * sin(val[0]);
                        }
    }
    info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'L', lnG, Hkk, lnG, dw); // Eigensolver of the BM model,zheevd for Hermitian matrix, highly efficient
    if (info != 0)
        cout << "BM_eigen error" << endl;
    for (int i = 0; i < lnG; i++) { // Sortation of energy, least first:check[0]=m least,check[1] passes m and picks the second least
        val[0] = 1000.0;
        for (int j = 0; j < lnG; j++) {
            Bool = 1;
            for (int i1 = 0; i1 < i; i1++)
                if (check[i1] == j) {
                    Bool = 0;
                    break;
                }
            if (Bool && val[0] > dw[j]) {
                val[0] = dw[j];
                check[i] = j;
            }
        }
    }
    for (int l = 0; l < 2; l++) { // Output energies and eigenvecotrs of two flat bands
        w[l] = dw[check[nG * 2 - 1 + l]];
        for (int j = 0; j < lnG; j++)
            vr[j * ncb + l] = Hkk[j * lnG + check[nG * 2 - 1 + l]]; // band1:vr[0,2,4];band2:vr[1,3,5]
    }
    if (ifbands)
        for (int l = 0; l < 20; l++) // Output energies for band plot with 20 bands
            w[l] = dw[check[nG * 2 - 10 + l]];
    delete[] Hkk;
}

double gamma1[4] = {0.18350342, 1.81649658, 1.81649658, 0.18350342},                          // Vallues of coefficients \gamma of auxiliary field
    gamma2[4] = {5.449489742783175, 0.550510257216822, 0.550510257216822, 5.449489742783175}, // Inverses
    ita[4] = {-3.301360247771569, -1.049295246550581, 1.049295246550581, 3.301360247771569};  // Vallues of \eta

void zmatsvd(MKL_Complex16 *A, double *S, MKL_Complex16 *U, MKL_Complex16 *V) { // SVD
    info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'A', 'A', lnk, lnk, A, lnk, S, U, lnk, V, lnk, superb);
    if (info != 0)
        cout << "LAPACKE_zgesvd error" << endl;
}

void zmatmul0(MKL_Complex16 a, MKL_Complex16 *B, MKL_Complex16 *C1) { // complex Matrix B times complex value a
    double val;
    for (int i = 0; i < Lnk; i++) {
        val = B[i].real;
        C1[i].real = val * a.real - B[i].imag * a.imag;
        C1[i].imag = val * a.imag + B[i].imag * a.real;
    }
}

void dmatmul(double a, MKL_Complex16 *B, MKL_Complex16 *C1) { // complex Matrix B times real value a
    for (int i = 0; i < Lnk; i++) {
        C1[i].real = a * B[i].real;
        C1[i].imag = a * B[i].imag;
    }
}

void zmatmul1(double *S, MKL_Complex16 *B) { // Diagonal matrix S times matrix B. Only diagonals of S stored.
    int Id4, Ia;
    for (Id4 = 0; Id4 < lnk; Id4++)
        for (Ia = 0; Ia < lnk; Ia++) {
            B[Id4 * lnk + Ia].real *= S[Id4];
            B[Id4 * lnk + Ia].imag *= S[Id4];
        }
}

void zmatmul2(MKL_Complex16 *B, double *S) { // matrix B times diagonal matrix S
    int Id4, Ia;
    for (Id4 = 0; Id4 < lnk; Id4++)
        for (Ia = 0; Ia < lnk; Ia++) {
            B[Ia * lnk + Id4].real *= S[Id4];
            B[Ia * lnk + Id4].imag *= S[Id4];
        }
}

void zmatmul3(MKL_Complex16 *A, MKL_Complex16 *B) { // Matrices multiplication. Store the result to the left matrix.
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lnk, lnk, lnk, &alpha, A, lnk, B, lnk, &beta, C, lnk);
    for (int Id4 = 0; Id4 < Lnk; Id4++)
        A[Id4] = C[Id4];
}

void zmatmul4(MKL_Complex16 *A, MKL_Complex16 *B) { // Matrices multiplication. Store the result to the right matrix.
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lnk, lnk, lnk, &alpha, A, lnk, B, lnk, &beta, C, lnk);
    for (int Id4 = 0; Id4 < Lnk; Id4++)
        B[Id4] = C[Id4];
}

void zmatcp(MKL_Complex16 *A, MKL_Complex16 *B) { // Matrices copy
    for (int Id4 = 0; Id4 < Lnk; Id4++)
        B[Id4] = A[Id4];
}

void zmati(MKL_Complex16 *A) {                                       // Matrix inverse
    info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, lnk, lnk, A, lnk, Ipiv); // First LU, the meaning of Ipiv?
    if (info != 0)
        cout << "Lapacke_zgetrf error" << endl;
    info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, lnk, A, lnk, Ipiv); // Then easy inverse
    if (info != 0)
        cout << "Lapacke_zgetri error" << endl;
}

void zmatadd(MKL_Complex16 *A, MKL_Complex16 *B) { // Matrix A add to B
    for (int Id4 = 0; Id4 < Lnk; Id4++) {
        B[Id4].real += A[Id4].real;
        B[Id4].imag += A[Id4].imag;
    }
}

void zmatinit(MKL_Complex16 *A) { // Make A an identity matrix
    int Id4;
    for (Id4 = 0; Id4 < Lnk; Id4++)
        A[Id4] = c0;
    for (Id4 = 0; Id4 < Lnk; Id4 += lnk + 1)
        A[Id4].real = 1.0;
}

void zmat_I(MKL_Complex16 *A) { // A substracts indentity matrix
    for (int Id4 = 0; Id4 < Lnk; Id4 += lnk + 1)
        A[Id4].real -= 1.0;
}

void zI_mat(MKL_Complex16 *A) { // Identity matrix substracts A
    int Id4;
    for (Id4 = 0; Id4 < Lnk; Id4++) {
        A[Id4].real *= -1.0;
        A[Id4].imag *= -1.0;
    }
    for (Id4 = 0; Id4 < Lnk; Id4 += lnk + 1)
        A[Id4].real += 1.0;
}

void zmataddI(MKL_Complex16 *A) { // A add indentity matrix
    for (int Id4 = 0; Id4 < Lnk; Id4 += lnk + 1)
        A[Id4].real += 1.0;
}

MKL_Complex16 zmatdet(MKL_Complex16 *A) {                            // Determinant of A by LU
    info = LAPACKE_mkl_zgetrfnp(LAPACK_ROW_MAJOR, lnk, lnk, A, lnk); // LU factorization
    if (info != 0)
        cout << "zmatdet Matrix error" << endl;
    MKL_Complex16 valz = alpha;
    double val;
    for (int Id4 = 0; Id4 < Lnk; Id4 += lnk + 1) { // Product of diagonals of upper triangle U:(a+ib)(c+id)(m+in)
        val = valz.real;
        valz.real = val * A[Id4].real - valz.imag * A[Id4].imag;
        valz.imag = val * A[Id4].imag + valz.imag * A[Id4].real;
    }
    return valz;
}

void zmatmaxmin(double *A, double *B) { // Split singularities to two digonal matrices. Only store diagonals.
    double valm;
    for (int Id4 = 0; Id4 < lnk; Id4++) {
        valm = B[Id4];
        if (valm > 1.0) {
            A[Id4] = valm;
            A[lnk + Id4] = 1.0;
        } else {
            A[lnk + Id4] = valm;
            A[Id4] = 1.0;
        }
    }
}

void eA(MKL_Complex16 *A) { // Exponentiation of matrix by eigenvalues and eigenvectors
    int i0, i1, i2;
    MKL_Complex16 *RM = new MKL_Complex16[Lnk],
                  *RN = new MKL_Complex16[Lnk],
                  ei[lnk];
    double val[3];
    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', lnk, A, lnk, ei, RM, lnk, RN, lnk); // Eigensolver of A
    if (info != 0)
        cout << "eA Matrix error" << endl;
    zmatcp(RN, A);
    zmati(A);
    for (i0 = 0; i0 < lnk; i0++) { // exp(a+ib)=exp(a)cos(b)+iexp(a)sin(b)
        val[0] = exp(ei[i0].real);
        val[1] = val[0] * sin(ei[i0].imag);
        val[0] *= cos(ei[i0].imag);
        for (i1 = 0; i1 < lnk; i1++) {
            i2 = i1 * lnk + i0;
            val[2] = RN[i2].real;
            RN[i2].real = val[2] * val[0] - RN[i2].imag * val[1];
            RN[i2].imag = val[2] * val[1] + RN[i2].imag * val[0]; //[RN(i1,i0).real+iRN(i1,i0)][expE.real+iexpE.imag]
        }
    }
    zmatmul4(RN, A); // RN exp(E) RN^-1
    delete[] RM;
    delete[] RN;
}

double Vx(double x) { // Coulumb interation in momentum space
    return Istrength * e * e * (1.0 - exp(-x * rp_unit * dgate)) * 0.5 / eps / (x * rp_unit);
}

void R(int ik, int nqG, double *MqG, MKL_Complex16 *_Mmn) {              // Exact single particle exitations for flat bands, our works 3, appendix D
    MKL_Complex16 M[ncb * ncb], wR[ncb], vlR[ncb * ncb], vrR[ncb * ncb]; // ncb=2 flat bands
    double val;
    for (int i = 0; i < ncb * ncb; i++)
        M[i] = c0;
    int I1, I2;
    for (int iqG = 0; iqG < nqG; iqG++) {
        val = Vx(MqG[iqG]) * 0.5 / Omega;
        for (int i1 = 0; i1 < ncb; i1++)
            for (int j1 = 0; j1 < ncb; j1++)
                for (int i2 = 0; i2 < ncb; i2++) {
                    I1 = ik * nqG * 4 + iqG * 4 + i1 * ncb + i2;
                    I2 = ik * nqG * 4 + iqG * 4 + j1 * ncb + i2;
                    M[i1 * ncb + j1].real += (_Mmn[I1].real * _Mmn[I2].real + _Mmn[I1].imag * _Mmn[I2].imag) * val;
                    M[i1 * ncb + j1].imag += (_Mmn[I1].real * _Mmn[I2].imag - _Mmn[I1].imag * _Mmn[I2].real) * val; //(_Mmn[I1].real-_Mmn[I1].imag i)(_Mmn[I2].real+_Mmn[I2].imag i)
                }
    }
    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', ncb, M, ncb, wR, vlR, ncb, vrR, ncb);
    if (info != 0)
        cout << "R error" << endl;
    else
        for (int i1 = 0; i1 < ncb; i1++)
            cout << wR[i1].real << "," << wR[i1].imag << " "; // output the eigenvalues of M
    cout << endl;
}

void Mat(int iqG, double val, MKL_Complex16 *Mmn, MKL_Complex16 *_Mmn, int nqG, int *kqG0, int *qqG, int *minusqG) { // Make matrices of density operator of each q+G and relate them to four values of auxiliary fields
    double v1;
    int iqG1;
    MKL_Complex16 *_Mmn1 = new MKL_Complex16[4 * Lnk], valm;
    for (int i = 0; i < 2 * Lnk; i++)
        _Mmn1[i] = c0;
    for (int l0 = 0; l0 < 2; l0++) { // dp_{q+G} and dp_{-q-G}
        if (l0 == 0)
            iqG1 = iqG;
        else
            iqG1 = minusqG[iqG];
        for (int I0 = 0; I0 < nk; I0++)
            for (int i = 0; i < ncb; i++)
                for (int j = 0; j < ncb; j++)
                    _Mmn1[l0 * Lnk + I0 * nk * 4 + i * lnk + kqG0[I0 * nqG + iqG1] * ncb + j] = _Mmn[I0 * nqG * 4 + iqG1 * 4 + i * ncb + j]; // nk0=3,nk=nk0*nk0,lnk=nk*ncb(ncb=2),Lnk=lnk*lnk
    }
    for (int l1 = 0; l1 < 2; l1++) { // Imag and real parts after decoupling
        v1 = -1.0 * val;
        valm.real = v1 * l1;
        valm.imag = v1 * (1 - l1);
        zmatmul0(valm, _Mmn1 + Lnk, _Mmn1 + 3 * Lnk);
        valm.real *= -1.0;
        zmatmul0(valm, _Mmn1, _Mmn1 + 2 * Lnk);
        zmatadd(_Mmn1 + 3 * Lnk, _Mmn1 + 2 * Lnk);
        zmatcp(_Mmn1 + 2 * Lnk, Mmn + l1 * Lnk); // Auxiliary field dependent density operator matrix stored in Mmn
    } // imag:A^2 part; real:b^2 part
    delete[] _Mmn1;
}

void SVD0(double *AF, double *expK, MKL_Complex16 *Mmn, MKL_Complex16 *UV, int nqG, MKL_Complex16 *B) { // Fisrt SVD over B matrices:B(nt,(n-1)t)...B(2t,1t)B(1t,0)
    MKL_Complex16 *Mmntmp = new MKL_Complex16[nqG * Lnk];
    MKL_Complex16 BR[4 * Lnk], RM[Lnk];
    double s[lnk];
    int I1, I2;
    for (int i = 0; i < lnk; i++)
        s[i] = 1.0;
    for (int i = 0; i < 4; i++)
        zmatinit(BR + i * Lnk);
    for (int itau = 0; itau < ntau; itau++) {
        for (int i = 0; i < Lnk; i++)
            RM[i] = c0;
        for (int i = 0; i < nqG; i++)
            dmatmul(AF[itau * nqG + i], Mmn + i * Lnk, Mmntmp + i * Lnk);
        for (int i = 0; i < nqG; i++)
            zmatadd(Mmntmp + i * Lnk, RM);
        eA(RM);
        zmatmul1(expK, RM); // consider H0
        zmatcp(RM, B + itau * Lnk);
        I1 = (itau % 2) * 2 * Lnk;
        zmatmul3(RM, BR + I1);
        zmatmul2(RM, s);
        I2 = ((itau + 1) % 2) * 2 * Lnk;
        zmatsvd(RM, s, BR + I2, BR + I2 + Lnk);
        zmatmul3(BR + I2 + Lnk, BR + I1 + Lnk);
        for (int i = 0; i < lnk; i++)
            S[itau * lnk + i] = s[i]; // Storing singular values
        for (int i = 0; i < 2 * Lnk; i++)
            UV[itau * 2 * Lnk + i] = BR[I2 + i]; // Sotring U(0<i<Lnk-1)V(Lnk<i<2Lnk-1)
    }
    delete[] Mmntmp;
}

void G00G0t(int itau, MKL_Complex16 *g00, MKL_Complex16 *g0t, MKL_Complex16 *UV) { // Calculate G(0,0) and G(0,t)
    int Id4;
    MKL_Complex16 *I1 = UV + itau * 2 * Lnk, *I2 = UV + (itau - 1) * 2 * Lnk; // I1:U(t)V(t);I2:U(t-1)V(t-1)
    zmatmaxmin(DR, S + (itau - 1) * lnk);                                     // DR=diag(S(t-1)max,S(t-1)min)
    zmatmaxmin(DL, S + itau * lnk);                                           // DL=diag(S(t)max,S(t)min)
    for (Id4 = 0; Id4 < Lnk; Id4++)
        RM[Id4] = c0;
    for (Id4 = 0; Id4 < lnk; Id4++)
        RM[Id4 * lnk + Id4].real = DR[lnk + Id4];
    zmatmul4(I2, RM);
    zmatmul4(I1 + Lnk, RM);
    zmatmul1(DL + lnk, RM); // RM=S(t)minV(t)U(t-1)S(t-1)min
    zmatcp(I1, RN);
    zmatmul4(I2 + Lnk, RN);
    zmati(RN); // RN=[V(t-1)U(t)]^{-1}
    for (Id4 = 0; Id4 < lnk; Id4++) {
        DL[Id4] = 1.0 / DL[Id4];
        DR[Id4] = 1.0 / DR[Id4];
    }
    zmatmul2(RN, DR);
    zmatmul1(DL, RN); // RN=S(t)max^{-1}[V(t-1)U(t)]^{-1}S(t-1)max^{-1}
    zmatadd(RN, RM);
    zmati(RM);
    zmatcp(I2 + Lnk, RN);
    zmati(RN);
    zmatmul2(RN, DR);
    zmatmul4(RN, RM); // RN=V(t-1)^{-1}S(t-1)max^{-1}
    zmatcp(I1, g00);
    zmati(g00);
    zmatmul1(DL, g00);
    zmatmul4(RM, g00); // g00=RM*S(t)max^{-1}U(t)^{-1}
    zmatcp(I1 + Lnk, g0t);
    zmatmul1(DL + lnk, g0t);
    zmatmul4(RM, g0t); // g0t=RM*S(t)min*V(t)
    for (Id4 = 0; Id4 < Lnk; Id4++) {
        g0t[Id4].real = -1 * g0t[Id4].real;
        g0t[Id4].imag = -1 * g0t[Id4].imag;
    }
}

void Gtt(int itau, MKL_Complex16 *gtt, MKL_Complex16 *gt0, MKL_Complex16 *UV) { // Calculate G(t,t) and G(t,0)
    int Id4;
    MKL_Complex16 *I1 = UV + itau * 2 * Lnk, *I2 = UV + (itau - 1) * 2 * Lnk;
    zmatmaxmin(DR, S + (itau - 1) * lnk);
    zmatmaxmin(DL, S + itau * lnk);
    for (Id4 = 0; Id4 < Lnk; Id4++)
        RM[Id4] = c0;
    for (Id4 = 0; Id4 < lnk; Id4++)
        RM[Id4 * lnk + Id4].real = DL[lnk + Id4];
    zmatmul4(I1, RM);
    zmatmul4(I2 + Lnk, RM);
    zmatmul1(DR + lnk, RM);
    zmatcp(I2, RN);
    zmatmul4(I1 + Lnk, RN);
    zmati(RN);
    for (Id4 = 0; Id4 < lnk; Id4++) {
        DL[Id4] = 1.0 / DL[Id4];
        DR[Id4] = 1.0 / DR[Id4];
    }
    zmatmul2(RN, DL);
    zmatmul1(DR, RN);
    zmatadd(RN, RM);
    zmati(RM);
    zmatcp(I1 + Lnk, RN);
    zmati(RN);
    zmatmul2(RN, DL);
    zmatmul4(RN, RM);
    zmatcp(I2, gtt);
    zmati(gtt);
    zmatmul1(DR, gtt);
    zmatmul4(RM, gtt);
    zmatcp(I2 + Lnk, gt0);
    zmatmul1(DR + lnk, gt0);
    zmatmul4(RM, gt0);
}

void SVD1(int itau, MKL_Complex16 *B1, MKL_Complex16 *UV, int idr, MKL_Complex16 *B2) { // SVD back and forth in mc update
    MKL_Complex16 *I1 = UV + itau * 2 * Lnk, *I2 = UV + (itau + direc[idr]) * 2 * Lnk;
    if (idr == 0) { // SVD to the left
        if (itau == 0)
            zmatsvd(B1, S, I1, I1 + Lnk);
        else {
            zmatmul3(B1, I2);
            zmatmul2(B1, S + (itau - 1) * lnk);
            zmatsvd(B1, S + itau * lnk, I1, I1 + Lnk);
            zmatmul3(I1 + Lnk, I2 + Lnk);
        }
        if (itau < ntau - 1) { // Prepare for next call of Gtt()
            I2 += 4 * Lnk;
            zmatmul3(B2, I1);
            zmatmul2(B2, S + itau * lnk);
            zmatsvd(B2, S + (itau + 1) * lnk, I2, I2 + Lnk);
            zmatmul3(I2 + Lnk, I1 + Lnk);
        }
    } else {                               // SVD to the right
        if (itau == 0 || itau == ntau - 1) // or
            zmatsvd(B1, S + itau * lnk, I1, I1 + Lnk);
        else {
            zmatmul4(I2 + Lnk, B1);
            zmatmul1(S + (itau + 1) * lnk, B1);
            zmatsvd(B1, S + itau * lnk, I1, I1 + Lnk);
            zmatmul4(I2, I1);
        }
    }
}

void zmattran(MKL_Complex16 *A, MKL_Complex16 *B) { // Calculate <C^+_iC_j> from <C_jC^+_i>
    int i0, i1, i2;
    for (i0 = 0; i0 < lnk; i0++)
        for (i1 = 0; i1 < lnk; i1++) {
            i2 = i0 * lnk + i1;
            B[i2].real = -A[i1 * lnk + i0].real;
            B[i2].imag = -A[i1 * lnk + i0].imag;
        }
    for (i0 = 0; i0 < Lnk; i0 += lnk + 1)
        B[i0].real += 1.0;
}

MKL_Complex16 SVP(MKL_Complex16 *gtt, MKL_Complex16 *RM) { // Calculate valley polirization
    MKL_Complex16 valz[2] = {c0, c0};
    zmattran(gtt, RM);
    int i0, i1, i2, i3, i4;
    for (i0 = 0; i0 < nk; i0++) // nk:number of momentum in mBZ
        for (i1 = 0; i1 < 2; i1++) {
            i2 = i0 * nk * 4 + i1 * lnk + i0 * 2 + i1; // lnk:2nk
            valz[0].real += RM[i2].real - gtt[i2].real;
            valz[0].imag += RM[i2].imag + gtt[i2].imag;
            for (i2 = 0; i2 < nk; i2++)
                for (i3 = 0; i3 < 2; i3++) {
                    i4 = i0 * nk * 4 + i1 * lnk + i2 * 2 + i3;
                    valz[1].real += gtt[i4].real * RM[i4].real - gtt[i4].imag * RM[i4].imag;
                    valz[1].imag += gtt[i4].real * RM[i4].imag + gtt[i4].imag * RM[i4].real;
                    valz[1].real += RM[i4].real * gtt[i4].real - RM[i4].imag * gtt[i4].imag;
                    valz[1].imag -= RM[i4].real * gtt[i4].imag + RM[i4].imag * gtt[i4].real;
                }
        }
    valz[1].real *= 1.0;
    valz[1].imag *= 1.0;
    valz[1].real += 2.0 * (valz[0].real * valz[0].real - valz[0].imag * valz[0].imag);
    valz[1].imag += 4.0 * valz[0].real * valz[0].imag;
    return valz[1];
}

MKL_Complex16 SIVC(MKL_Complex16 *gtt, MKL_Complex16 *RM) { // Calculate intervalley coherance
    MKL_Complex16 valz2[2] = {c0, c0};
    zmattran(gtt, RM);
    int i0, i1, i2, i3, i4, i5;
    int direc2[2] = {1, 0};     // For IVC
    for (i0 = 0; i0 < nk; i0++) // nk:number of momentum in mBZ
        for (i1 = 0; i1 < 2; i1++) {
            for (i2 = 0; i2 < nk; i2++)
                for (i3 = 0; i3 < 2; i3++) {
                    i4 = i0 * nk * 4 + i1 * lnk + i2 * 2 + direc2[i3];
                    i5 = i0 * nk * 4 + i1 * lnk + i2 * 2 + direc2[i3];
                    valz2[1].real += RM[i4].real * RM[i5].real + RM[i4].imag * RM[i5].imag;
                    valz2[1].imag += -RM[i4].real * RM[i5].imag + RM[i4].imag * RM[i5].real;
                    valz2[1].real += gtt[i4].real * gtt[i5].real + gtt[i4].imag * gtt[i5].imag;
                    valz2[1].imag += gtt[i4].real * gtt[i5].imag - gtt[i4].imag * gtt[i5].real;
                }
        }
    valz2[1].real *= 1.0;
    valz2[1].imag *= 1.0;
    return valz2[1];
}

int npbc(int nr, int nk0) {
    if (nr >= nk0)
        return nr - nk0;
    if (nr < 0)
        return nr + nk0;
    return nr;
}

MKL_Complex16 SIVCq(MKL_Complex16 *gtt, MKL_Complex16 *RM, int i, int j) { // Calculate intervalley coherance
    MKL_Complex16 valz2[2] = {c0, c0};
    zmattran(gtt, RM);
    int kx1, ky1, kx2, ky2, n1, n2, tempk1, tempk2, tempk1q, tempk2q, qx, qy, i0, i1;
    qx = i;
    qy = j;
    for (kx1 = 0; kx1 < nk0; kx1++) {
        for (ky1 = 0; ky1 < nk0; ky1++) {
            for (kx2 = 0; kx2 < nk0; kx2++) {
                for (ky2 = 0; ky2 < nk0; ky2++) {
                    for (n1 = 0; n1 < 2; n1++) {
                        for (n2 = 0; n2 < 2; n2++) {
                            tempk1 = (kx1 * nk0 + ky1) * 2 + n1;
                            tempk2 = (kx2 * nk0 + ky2) * 2 + 1 - n2;
                            tempk1q = (npbc(kx1 + qx, nk0) * nk0 + npbc(ky1 + qy, nk0)) * 2 + n1;
                            tempk2q = (npbc(kx2 + qx, nk0) * nk0 + npbc(ky2 + qy, nk0)) * 2 + 1 - n2;
                            i0 = tempk1 * lnk + tempk2;
                            i1 = tempk1q * lnk + tempk2q;
                            valz2[1].real += RM[i0].real * RM[i1].real + RM[i0].imag * RM[i1].imag;
                            valz2[1].imag += -RM[i0].real * RM[i1].imag + RM[i0].imag * RM[i1].real;
                            valz2[1].real += gtt[i0].real * gtt[i1].real + gtt[i0].imag * gtt[i1].imag;
                            valz2[1].imag += gtt[i0].real * gtt[i1].imag - gtt[i0].imag * gtt[i1].real;
                        }
                    }
                }
            }
        }
    }
    return valz2[1];
}

MKL_Complex16 utSIVC(MKL_Complex16 *gt0, MKL_Complex16 *g0t, int i, int j) { // Calculate unequal-time intervalley coherance
    MKL_Complex16 valz2[2] = {c0, c0};
    int kx1, ky1, kx2, ky2, n1, n2, tempk1, tempk2, tempk1q, tempk2q, qx, qy, i0, i1, i2, i3;
    qx = i;
    qy = j;
    for (kx1 = 0; kx1 < nk0; kx1++) {
        for (ky1 = 0; ky1 < nk0; ky1++) {
            for (kx2 = 0; kx2 < nk0; kx2++) {
                for (ky2 = 0; ky2 < nk0; ky2++) {
                    for (n1 = 0; n1 < 2; n1++) {
                        for (n2 = 0; n2 < 2; n2++) {
                            tempk1 = (kx1 * nk0 + ky1) * 2 + n1;
                            tempk2 = (kx2 * nk0 + ky2) * 2 + 1 - n2;
                            tempk1q = (npbc(kx1 + qx, nk0) * nk0 + npbc(ky1 + qy, nk0)) * 2 + n1;
                            tempk2q = (npbc(kx2 + qx, nk0) * nk0 + npbc(ky2 + qy, nk0)) * 2 + 1 - n2;
                            i0 = tempk2 * lnk + tempk1;
                            i1 = tempk2q * lnk + tempk1q;
                            i2 = tempk1 * lnk + tempk2;
                            i3 = tempk1q * lnk + tempk2q;
                            valz2[1].real += g0t[i0].real * g0t[i1].real + g0t[i0].imag * g0t[i1].imag;
                            valz2[1].imag += -g0t[i0].real * g0t[i1].imag + g0t[i0].imag * g0t[i1].real;
                            valz2[1].real += gt0[i2].real * gt0[i3].real + gt0[i2].imag * gt0[i3].imag;
                            valz2[1].imag += gt0[i2].real * gt0[i3].imag - gt0[i2].imag * gt0[i3].real;
                        }
                    }
                }
            }
        }
    }
    return valz2[1];
}

int main() {
    int rank, size;
    MPI_Init(0, 0); // Beginning of paralleling interface
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::random_device seed; // Random seed
    std::ranlux48 engine(seed());
    std::normal_distribution<double> uni1(0.0, 1.0);
    std::uniform_int_distribution<unsigned> uni2(0, 10000); // Uniform random integers with boundaries included
    double val[8],
        k0[nk * 2],     // k points smearing in mBZ
        q0[nk * 2],     // Momentum transfer for Coulumb interaction
        qG[nk * 5 * 2], // Add some shifts   why 5?
        MqG[nk * 5],    // Module of qG
        G2[25 * 2];     // Combination of lattices of mBZ to be used as shifts
    int nqG = 0,        // Number of qG within a cutoff
        Bool,
        Ia, Id0, Id1, Id2, Id3, Id4, // Some integer variables
        kqG0[nk * nk * 5],           // Index of k in k+q+G
        kqG[nk * nk * 5],            // Index of G in k+q+G
        qqG[nk * 25],                // Index of q in q+G
        IG[25 * nG],                 // Index of equivalent G in G2+G
        minusqG[nk * 5];             // Index of -q-G in the order of q+G

    int n_dot = nk0 * 3 / 2;

    int q1[n_dot];
    int q2[n_dot];

    if (nk0 == 9) {
        int q1_tmp[] = {6,5,4,5,2,1,0,0,0,0,0,1,3};
        int q2_tmp[] = {3,1,0,0,0,0,0,1,2,3,4,5,6};
        for (int i = 0; i < n_dot; ++i) {
            q1[i] = q1_tmp[i];
            q2[i] = q2_tmp[i];
        }
    }
    
    if (nk0 == 6) {
        int q1_tmp[] = {4,3,2,1,0,0,0,0,3};
        int q2_tmp[] = {3,0,0,0,0,1,2,3,4};
        for (int i = 0; i < n_dot; ++i) {
            q1[i] = q1_tmp[i];
            q2[i] = q2_tmp[i];
        }
    }

    // open
    std::ifstream inputFile("pin.in");
    // read
    inputFile >> u0;
    inputFile >> u1;
    inputFile >> dgate;
    inputFile >> Istrength;
    inputFile >> Tstart;

    // close
    inputFile.close();
    // u0 = 0.06;
    // u1 = 0.11;
    // dgate = 40e-9;
    // Istrength = 1;
    // Tstart = 3;

    u0 = u0 * e;
    u1 = u1 * e;

    // construct file name
    std::ostringstream filename;
    filename << "CFMC" << nk0 << ".dat";

    for (int i = 0; i < nk0; i++)
        for (int j = 0; j < nk0; j++)
            for (int l = 0; l < 2; l++) {
                k0[(i * nk0 + j) * 2 + l] = G0[0][l] / nk0 * (i + 0.0) + G0[1][l] / nk0 * (j + 0.0); // Smearing in mBZ
                q0[(i * nk0 + j) * 2 + l] = G0[0][l] / nk0 * i + G0[1][l] / nk0 * j;                 // Similar smearing
            }

    Bool = 0;
    for (int i = -_nG; i < _nG + 1; i++) // Momenta in the extended mBZ for BM model
        for (int j = -_nG; j < _nG + 1; j++) {
            val[0] = 0.0;
            for (int l = 0; l < 2; l++)
                val[0] += pow(G0[0][l] * i + G0[1][l] * j, 2);
            if (val[0] < BMcut) {
                for (int l = 0; l < 2; l++)
                    G[Bool * 2 + l] = G0[0][l] * i + G0[1][l] * j;
                Bool += 1;
            }
        }
    if (Bool != nG)
        cout << "G error" << endl;

    for (int i = -2; i < 3; i++)
        for (int j = -2; j < 3; j++)
            for (int l = 0; l < 2; l++)
                G2[((i + 2) * 5 + j + 2) * 2 + l] = G0[0][l] * i + G0[1][l] * j;
    for (int i = 0; i < 25; i++) // Make shift and find equivalent,form factor,the index of G+G'
        for (int j = 0; j < nG; j++) {
            for (int l = 0; l < 2; l++)
                val[l] = G2[i * 2 + l] + G[j * 2 + l];
            Bool = 1;
            for (int j1 = 0; j1 < nG; j1++)
                if (abs1(val[0] - G[j1 * 2]) < 1.0e-8 && abs1(val[1] - G[j1 * 2 + 1]) < 1.0e-8) { // To find equivalent G after shifting
                    IG[i * nG + j] = j1;                                                          // Label the found equivalent
                    Bool = 0;
                    break;
                }
            if (Bool)
                IG[i * nG + j] = -1; // If no equivalent, label -1
        }

    for (int i = 0; i < nk; i++) // q+G
        for (int j = 0; j < 25; j++) {
            for (int l = 0; l < 2; l++)
                val[l] = q0[i * 2 + l] + G2[j * 2 + l];
            val[2] = pow(val[0], 2) + pow(val[1], 2);
            if (val[2] > 1.0e-8 && val[2] < 3.01) { // Cut off of q +G at the module of a lattice of mBZ. 0 exclusive.
                for (int l = 0; l < 2; l++)
                    qG[nqG * 2 + l] = val[l];
                qqG[nqG] = i;
                MqG[nqG] = sqrt(val[2]);
                nqG += 1;
            }
        }

    for (int i1 = 0; i1 < nqG; i1++) {
        Bool = 1;
        for (int i2 = 0; i2 < nqG; i2++)
            if (abs1(qG[i2 * 2] + qG[i1 * 2]) < 1.0e-8 && abs1(qG[i2 * 2 + 1] + qG[i1 * 2 + 1]) < 1.0e-8) {
                minusqG[i1] = i2;
                Bool = 0;
                break;
            }
        if (Bool) {
            cout << "minusqG error" << endl;
            exit(0);
        }
    }

    for (int i = 0; i < nk; i++)        // k
        for (int j = 0; j < nqG; j++) { // qG
            for (int l1 = 0; l1 < 2; l1++)
                val[l1] = k0[i * 2 + l1] + qG[j * 2 + l1]; // k+q+G
            Bool = 1;
            for (int l1 = 0; l1 < nk; l1++) {
                for (int l2 = 0; l2 < 25; l2++)
                    if (abs1(val[0] - G2[l2 * 2] - k0[l1 * 2]) < 1.0e-8 && abs1(val[1] - G2[l2 * 2 + 1] - k0[l1 * 2 + 1]) < 1.0e-8) { // the index of (k+q) and G2
                        kqG0[i * nqG + j] = l1;
                        kqG[i * nqG + j] = l2;
                        Bool = 0;
                        break;
                    }
                if (Bool == 0)
                    break;
            }
            if (Bool) {
                cout << "kqG error" << endl;
                exit(0);
            }
        }

    MKL_Complex16 *_Mmn = new MKL_Complex16[nk * nqG * 4]; // To store form factors:lambda(m1,m2,k+q+G); bands:2*2; k:nk; q+G:nqG
    ofstream outf;                                         // Output class
    ifstream inf;
    double tE[lnk + 2];                                    // To store kenetics
    if (rank == 0) {
        MKL_Complex16 *vr = new MKL_Complex16[nk * lnG * ncb]; // To store eigenvectors of BM for each k (number:nk), with lnG for X, ncb for mi;
        outf.open("data.dat", ios::out);                       // Output data file
        double k1[2], dk[2],
            kpath[5][2] = {{0.0, 0.0}, {sqrt(3.0) / 2.0, 0.0}, {sqrt(3.0) / 2.0, -0.5}, {0.0, -1.0}, {0.0, 0.0}}; // High symmetry path
        for (int i = 0; i < 4; i++) {
            for (int l = 0; l < 2; l++)
                dk[l] = (kpath[i + 1][l] - kpath[i][l]) / 20.0; // Split each sector to 20 slices
            for (int j = 0; j < 20; j++) {
                for (int l = 0; l < 2; l++)
                    k1[l] = kpath[i][l] + dk[l] * j;
                BM_eigen(k1, tE, vr, 1); // To solve for bands
                for (int l = 0; l < 20; l++)
                    outf << tE[l] << " "; // Output to the first line of data.dat
            }
        }
        outf << endl; // Change line
        outf.close();

        for (int ik = 0; ik < nk; ik++)                                 // Calculate eigenvalues and eigenvectors of k points smearing the mBZ.
            BM_eigen(k0 + ik * 2, tE + ik * 2, vr + ik * lnG * ncb, 0); // a k0 has kx and ky, outputs 2tE and 2 vr;vr[0,2,4],vr[1,3,5] represents 2 bands
        for (int i = 0; i < nk; i++)                                    // Calculate form factor for each k
            for (int j = 0; j < nqG; j++)                               // With different q+G
                for (int i1 = 0; i1 < ncb; i1++)                        // Only flat bands are kept
                    for (int j1 = 0; j1 < ncb; j1++) {
                        Ia = (i * nqG + j) * 4 + i1 * 2 + j1; // Indices for storing:lambda(ik+j(q+G),i1 band,j1 band)
                        _Mmn[Ia] = c0;
                        for (int l = 0; l < nG; l++) {
                            Id0 = IG[kqG[i * nqG + j] * nG + l]; // Index of G in k+q+G, plus l G'
                            if (Id0 > -1)
                                for (int l2 = 0; l2 < 4; l2++) {
                                    Id2 = i * lnG * ncb + (l * 4 + l2) * ncb + i1;                              // u*(ik,lG',l2X,i1 band)
                                    Id3 = kqG0[i * nqG + j] * lnG * ncb + (Id0 * 4 + l2) * ncb + j1;            // u(index of (k+q) in k+q+G,Index of G in k+q+G plus l G',l2X,j1 band)
                                    _Mmn[Ia].real += vr[Id2].real * vr[Id3].real + vr[Id2].imag * vr[Id3].imag; //_Mmn[nk*nqG*4]; k:nk; nqG:q+G; 4:2band*2band
                                    _Mmn[Ia].imag += vr[Id2].real * vr[Id3].imag - vr[Id2].imag * vr[Id3].real;
                                }
                        }
                    }
        delete[] vr;
    }
    MPI_Bcast(tE, lnk, MPI_DOUBLE, 0, MPI_COMM_WORLD);                      // Copy from root thread to other threads
    MPI_Bcast(_Mmn, nk * nqG * 4, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); // all cores share the same tE and _Mmn

    double CN[2],           // To store VP
        CN2[2],             // To store IVC
        FCN[3],             // To store sum over all threads
        FCN2[3],            // To store sum over all threads
        kBT,                // Boltzmann constant times temperature
        dtau;               // Imaginary time slice
    MKL_Complex16 gtt[Lnk], // To store equal time Green's function G(t,t)
        gt0[Lnk],           // To store nonequal time G(t,0)
        g00[Lnk],           // To store equal time Green's function G(0,0)
        g0t[Lnk],           // To store nonequal time G(0,t)
            *b = NULL,      // empty thread
        b1[Lnk],
            B1[Lnk],
            valz,
            valz2,
            valzq,
            valzut,
            CNut[(ntau + 1)],
            CNut2[(ntau + 1)],
            CNut3[(ntau + 1)],

            SIVCqt[n_dot*(ntau+1)],

            *fgt0 = new MKL_Complex16[(ntau + 1) * lnk]; // To store two eigenvalues of the diagonal blocks
    int NumAF = ntau * nqG,                              // The number of auxiliary fieds
        CqG,                                             // The number of half of q+G
        itau,
        idr;

    double *gAF = NULL, // The gathering of auxiliary fields of all threads
        *AF = new double[NumAF],
           AFtmp[nqG]; // Auxiliary fields
    if (rank == 0)
        gAF = new double[size * NumAF];
    for (int i = 0; i < lnk; i++)
        S[ntau * lnk + i] = 1.0;

    if (rank == 0) {
        cout << "Start R" << endl;
        for (int i = 0; i < nk; i++)
            R(i, nqG, MqG, _Mmn); // solve single-particle excitation
        cout << "end R" << endl;
    }

    std::vector<double> iT_values = {0.667};
    for (double iT : iT_values) {
        MKL_Complex16 *Mmn = new MKL_Complex16[nqG * Lnk],             // To store density matrices A^2 and B^2
            *B = new MKL_Complex16[ntau * Lnk],                        // B matrix of each time slice
                *UV = new MKL_Complex16[(ntau + 1) * 2 * Lnk],         // To store UV of each time slice
                    *UVtmp = new MKL_Complex16[(ntau + 1) * 2 * Lnk],  // To store UV before unequal-time update
                        *Gt0 = new MKL_Complex16[(ntau + 1) * nk * 4]; // To store diagonal blocks of G(t,0)
        double *expK = new double[lnk];                                // To store kinetic term

        for (int i = 0; i < 2; i++)
            zmatinit(UV + ntau * 2 * Lnk + i * Lnk);

        kBT = 1.0e-3 * e * iT;   // Units in meV
        
        dtau = 1.0 / kBT / ntau; // Split imaginary time

        for (int i = 0; i < lnk; i++) // get H0
            expK[i] = std::exp(-dtau * tE[i]);

        CqG = 0;
        for (int i = 0; i < nqG; i++)
            if ((abs1(qG[i * 2 + 1]) < 1.0e-8 && qG[i * 2] > 1.0e-8) || qG[i * 2 + 1] < -1.0e-8) { // Lower half of q+G
                Mat(i, sqrt(dtau * Vx(MqG[i]) * 0.5 / Omega), Mmn + CqG * 2 * Lnk, _Mmn, nqG, kqG0, qqG, minusqG);
                CqG += 1;
            }
        /* Observable initial */
        for (int i = 0; i < 2; i++)
            CN[i] = 0.0;
        for (int i = 0; i < 2; i++)
            CN2[i] = 0.0;
        for (int i = 0; i < (ntau + 1) * lnk; i++)
            fgt0[i] = c0;
        for (int i = 0; i < (ntau + 1); i++)
            CNut[i] = c0;
        for (int i = 0; i < (ntau + 1); i++)
            CNut2[i] = c0;
        for (int i = 0; i < (ntau + 1); i++)
            CNut3[i] = c0;
        for (int i = 0; i < (ntau + 1) * nk * 4; i++)
            Gt0[i] = c0;
        for (int i = 0; i < (ntau + 1) * n_dot; i++)
            SIVCqt[i] = c0;

        if (rank == 0) { // Initiate auxiliary fields by random integers for all threads
            for (int i = 0; i < size * NumAF; i++)
                gAF[i] = uni1(engine); // normal distribution?
            if (ifcontinue) {          // If continue from last run
                inf.open("gAF.dat", ios::in);
                for (int i = 0; i < size * NumAF; i++)
                    inf >> gAF[i]; // Assign last fields
                inf.close();
            }
        }

        MPI_Scatter(gAF, NumAF, MPI_DOUBLE, AF, NumAF, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Distribute gAF[NumAF*i,NumAF*(i+1)-1] to the AF in rank i
        if (rank == 0) {
            outf.open("AF.dat", ios::out); // Output all auxiliary fields
            for (int i = 0; i < NumAF; i++)
                outf << AF[i] << " ";
            outf.close();
        }
        SVD0(AF, expK, Mmn, UV, nqG, B);

        
        double r = 0.1,
               deltar = 0.1,
               acrate = 0.0,
               acnum = 0.0;
        MKL_Complex16 *Mmntmp1 = new MKL_Complex16[nqG * Lnk],
                      Mmntmp2[Lnk],
                      Mmntmp3[Lnk],
                      Mmntmp4[Lnk];
        double sum1,
            c1,
            sum2,
            c2;

        // ofstream acceptrate;
        // acceptrate.open("acceptrate.dat");


        int measure_times = 0;
        for (int imc = 0; imc < Nmc; imc++) {
            if (imc % 50 == 0)
                cout << "rank " << rank << " imc " << imc << endl;
            if (imc > 0)
                acrate = acnum / (2 * ntau * imc);
            if (imc > 0)
                r += deltar * (acrate - 0.4);
            if (rank == 0) {
                outf.open("accrate.dat", ios::app);
                outf << "imc" << imc << " " << "acrate" << acrate;
                outf << endl;
                outf.close();
            }

            for (idr = 1; idr > -1; idr--) { // Back and forth update over imaginary time;idr=1,from B((n-1)t,(n-2)t) to B(t,0);idr=0,from B(t,0) to B((n-1)t,(n-2)t);

                if (idr == 1 &&((ifcontinue == 0 && imc >= Smc)|| ifcontinue == 1 && imc >= 1)) {
                    measure_times += 1;
                    // Copy UV and S
                    for (int i = 0; i < (ntau + 1) * lnk; i++)
                        Stmp[i] = S[i];
                    for (int i = 0; i < (ntau + 1) * 2 * Lnk; i++)
                        UVtmp[i] = UV[i];

                    for (int itau = ntau - 1; itau >= 0; --itau) { // measure unequal-time
                        Gtt(itau + 1, gtt, gt0, UV);
                        G00G0t(itau + 1, g00, g0t, UV);

                        for (int q_i = 0; q_i < n_dot; q_i++){

                            if (itau == ntau - 1) {         
                                valzq = SIVCq(gtt, RM, q1[q_i], q2[q_i]);
                                SIVCqt[0 * n_dot + q_i].real += valzq.real;
                                SIVCqt[0 * n_dot + q_i].imag += valzq.imag;
                            }
                            valzut = utSIVC(gt0, g0t, q1[q_i], q2[q_i]);
                            SIVCqt[(itau+1) * n_dot + q_i].real += valzut.real;
                            SIVCqt[(itau+1) * n_dot + q_i].imag += valzut.imag;
                        }

                        // measure <S(t)S(0)>
                        if (itau == ntau - 1) {
                            valzq = SIVCq(gtt, RM, 0, 0);
                            CNut[0].real += valzq.real;
                            CNut[0].imag += valzq.imag;
                        }
                        valzut = utSIVC(gt0, g0t, 0, 0);
                        CNut[itau + 1].real += valzut.real;
                        CNut[itau + 1].imag += valzut.imag;

                        if (itau == ntau - 1) {
                            valzq = SIVCq(gtt, RM, 0, 1);
                            CNut2[0].real += valzq.real;
                            CNut2[0].imag += valzq.imag;
                        }
                        valzut = utSIVC(gt0, g0t, 0, 1);
                        CNut2[itau + 1].real += valzut.real;
                        CNut2[itau + 1].imag += valzut.imag;

                        if (itau == ntau - 1) {
                            valzq = SIVCq(gtt, RM, 1, 0);
                            CNut3[0].real += valzq.real;
                            CNut3[0].imag += valzq.imag;
                        }
                        valzut = utSIVC(gt0, g0t, 1, 0);
                        CNut3[itau + 1].real += valzut.real;
                        CNut3[itau + 1].imag += valzut.imag;

                        // measure G(t,0)
                        for (Id0 = 0; Id0 < nk; Id0++) // Diagonal blocks of G(t,0)
                            for (Id1 = 0; Id1 < 2; Id1++)
                                for (Id2 = 0; Id2 < 2; Id2++) {
                                    Id3 = (itau + 1) * nk * 4 + Id0 * 4 + Id1 * 2 + Id2;
                                    Id4 = Id0 * nk * 4 + Id1 * lnk + Id0 * 2 + Id2;
                                    Gt0[Id3].real += gt0[Id4].real;
                                    Gt0[Id3].imag += gt0[Id4].imag;
                                    if (itau == ntau - 1) { // G(beta,0)=G(beta,beta)=G(0,0)
                                        Id3 = Id0 * 4 + Id1 * 2 + Id2;
                                        Gt0[Id3].real += gtt[Id4].real;
                                        Gt0[Id3].imag += gtt[Id4].imag;
                                    }
                                }

                        zmatcp(B + itau * Lnk, B1);
                        zmatinit(RM);                // will not be used in SVD1
                        SVD1(itau, B1, UV, idr, RM); // SVD back
                    }

                    // give back UV and S
                    for (int i = 0; i < (ntau + 1) * lnk; i++)
                        S[i] = Stmp[i];
                    for (int i = 0; i < (ntau + 1) * 2 * Lnk; i++)
                        UV[i] = UVtmp[i];
                }

                for (itau = idr * (ntau - 1); itau != (1 - idr) * ntau - idr; itau += 1 - 2 * idr) { // idr=1,ntau-1 to 0;idr=0,0 to ntau-1
                    Gtt(itau + 1, gtt, gt0, UV);
                    if (imc >= Smc && idr == 1) { // Measurement
                        valz = SVP(gtt, RM);
                        CN[0] += valz.real;
                        CN[1] += valz.imag;
                        valz2 = SIVC(gtt, RM);
                        CN2[0] += valz2.real;
                        CN2[1] += valz2.imag;
                    }

                    // update field in itau
                    for (Id0 = 0; Id0 < nqG; Id0++) {
                        AFtmp[Id0] = AF[itau * nqG + Id0];
                    }
                    for (Id0 = 0; Id0 < nqG; Id0++) {
                        AFtmp[Id0] += r * uni1(engine);
                    }

                    // get B'(itau)
                    for (Id0 = 0; Id0 < nqG; Id0++) {
                        dmatmul(AFtmp[Id0], Mmn + Id0 * Lnk, Mmntmp1 + Id0 * Lnk);
                    }
                    for (int i = 0; i < Lnk; i++) {
                        Mmntmp2[i] = c0;
                    }
                    for (Id0 = 0; Id0 < nqG; Id0++) {
                        zmatadd(Mmntmp1 + Id0 * Lnk, Mmntmp2);
                    }
                    eA(Mmntmp2);
                    zmatmul1(expK, Mmntmp2); // B'(tau) with H0

                    // calculate delta
                    for (int i = 0; i < Lnk; i++) {
                        Mmntmp3[i] = B[itau * Lnk + i];
                    }
                    // copy the old B(itau+1,itau)
                    for (int i = 0; i < Lnk; i++) {
                        Mmntmp4[i] = B[itau * Lnk + i];
                    }
                    zmati(Mmntmp3); // B^-1(tau)
                    zmatcp(Mmntmp2, RN);
                    zmatmul3(RN, Mmntmp3);
                    zmat_I(RN); // RN=B'(tau)*B^-1(tau)-I=Delta

                    // calculate ratio
                    zmatcp(gtt, RM);
                    zI_mat(RM);       // RM=I-gtt
                    zmatmul3(RN, RM); // RN=Delta*(I-gtt)
                    zmataddI(RN);     // RN=I+Delta*(I-gtt)
                    valz = zmatdet(RN);
                    val[1] = square(valz.real * valz.real + valz.imag * valz.imag); // fermion part ratio

                    sum1 = 0.0;
                    for (int i = 0; i < nqG; i++) {
                        sum1 += AFtmp[i] * AFtmp[i];
                    }
                    c1 = exp(-0.5 * sum1); // the new coefficient

                    sum2 = 0.0;
                    for (int i = 0; i < nqG; i++) {
                        sum2 += AF[itau * nqG + i] * AF[itau * nqG + i];
                    }
                    c2 = exp(-0.5 * sum2); // the old coefficient

                    val[1] *= c1;
                    val[1] /= c2; // the real ratio

                    // acceptrate << val[1] << endl;



                    if (uni2(engine) * 1.0e-4 <= val[1]) { // Accept
                        for (Id0 = 0; Id0 < nqG; Id0++) {
                            AF[itau * nqG + Id0] = AFtmp[Id0];
                        }
                        zmatcp(Mmntmp2, B + itau * Lnk);
                        acnum += 1;
                        zmatcp(Mmntmp2, B1);
                    } else {
                        zmatcp(Mmntmp4, B1);
                    }
                    // else: do nothing (reject)

                    if (idr == 0 && itau < ntau - 1)
                        zmatcp(B + (itau + 1) * Lnk, RM);
                    SVD1(itau, B1, UV, idr, RM); // SVD back and forth
                } // time loop
            } // direction loop

            if (imc >= Smc && (imc - Smc + 1) % 2 == 0) { // Obtain eigenvalues of diagonal blocks of G(t,0) every 2 mc steps
                for (Id0 = 0; Id0 < (ntau + 1) * nk; Id0++) {
                    info = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'N', 'V', 2, Gt0 + Id0 * 4, 2, C, RM, 2, RN, 2);
                    if (info != 0)
                        cout << "LAPACKE_zgeev error" << endl;
                    // if (C[0].real > C[1].real) {
                    // std::swap(C[0], C[1]);
                    // }
                    for (Id1 = 0; Id1 < 2; Id1++) {
                        fgt0[Id0 * 2 + Id1].real += C[Id1].real;
                        fgt0[Id0 * 2 + Id1].imag += C[Id1].imag;
                    }
                }
                for (Id0 = 0; Id0 < (ntau + 1) * nk * 4; Id0++) // Clean for new cumulation
                    Gt0[Id0] = c0;
            }

        } // MC loop
        // acceptrate.close();
        delete[] Mmntmp1;
        delete[] Mmn;
        delete[] B;
        delete[] UV;
        delete[] UVtmp;
        delete[] Gt0;
        delete[] expK;

        int measure_pre = 0;
        if (rank == 0){
            inf.open("measure_times.dat");
            if (inf.is_open()){
                inf >> measure_pre;
            } else{
                measure_pre = 0;
            }
            inf.close();

            outf.open("measure_times.dat");
            outf << measure_times + measure_pre << endl;
            outf.close();
        }

        //val[0] = (Nmc - Smc) * ntau * nk * nk;

        for (int i = 0; i < (ntau + 1) * lnk; i++) {
            fgt0[i].real /= (Nmc - Smc); // Final nonequal-time Green's function for each k point of two bands
            fgt0[i].imag /= (Nmc - Smc);
        }

        for (int i = 0; i < (ntau + 1); i++) {
            CNut[i].real /= (Nmc - Smc) * nk * nk; // Final nonequal-time SIVC
            CNut[i].imag /= (Nmc - Smc) * nk * nk;
        }
        for (int i = 0; i < (ntau + 1); i++) {
            CNut2[i].real /= (Nmc - Smc) * nk * nk; // Final nonequal-time SIVC
            CNut2[i].imag /= (Nmc - Smc) * nk * nk;
        }
        for (int i = 0; i < (ntau + 1); i++) {
            CNut3[i].real /= (Nmc - Smc) * nk * nk; // Final nonequal-time SIVC
            CNut3[i].imag /= (Nmc - Smc) * nk * nk;
        }
        for (int i = 0; i < (ntau+1)*n_dot; i++){
            SIVCqt[i].real /= nk * nk;
            SIVCqt[i].imag /= nk * nk;
        }

        CN[0] /= 1.0*(Nmc - Smc) * ntau * nk * nk; // Final value of valley polarization
        CN[1] /= 1.0*(Nmc - Smc) * ntau * nk * nk;
        CN2[0] /= 1.0*(Nmc - Smc) * ntau * nk * nk; // Final value of intervalley coherance
        CN2[1] /= 1.0*(Nmc - Smc) * ntau * nk * nk;

        MKL_Complex16 *Fgt0 = NULL;
        if (rank == 0)
            Fgt0 = new MKL_Complex16[(size + 1) * (ntau + 1) * lnk];
        MPI_Gather(fgt0, (ntau + 1) * lnk, MPI_C_DOUBLE_COMPLEX, Fgt0, (ntau + 1) * lnk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); // Gather fgt0 of all
        if (rank == 0) {
            outf.open("SACugt.dat", ios::app);
            for (int k = 0; k < lnk; k += 2) { 
                outf << "momentum" << k << " (smaller of k=" << k << " and k=" << k + 1 << ")" << endl;
                for (int i = 0; i < size; i++) {
                    outf << "rank" << i << endl;
                    for (int j = 0; j < (ntau + 1); j++) {
                        int Id0 = i * (ntau + 1) * lnk + j * lnk + k;
                        int Id1 = Id0 + 1; 
                        double val0 = Fgt0[Id0].real;
                        double val1 = Fgt0[Id1].real;
                        outf << std::setprecision(3) << std::min(val0, val1) << " ";
                    }
                    outf << endl;
                }

                outf << "momentum" << k + 1 << " (larger of k=" << k << " and k=" << k + 1 << ")" << endl;
                for (int i = 0; i < size; i++) {
                    outf << "rank" << i << endl;
                    for (int j = 0; j < (ntau + 1); j++) {
                        int Id0 = i * (ntau + 1) * lnk + j * lnk + k;
                        int Id1 = Id0 + 1;
                        double val0 = Fgt0[Id0].real;
                        double val1 = Fgt0[Id1].real;
                        outf << std::setprecision(3) << std::max(val0, val1) << " ";
                    }
                    outf << endl;
                }
            }
            outf.close();
        }

        MPI_Reduce(fgt0, Fgt0, (ntau + 1) * lnk, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); // Sum them over all threads
        for (int i = 0; i < (ntau + 1) * lnk; i++) {
            fgt0[i].real = fgt0[i].real * fgt0[i].real;
            fgt0[i].imag = 0.0;
        }
        MPI_Reduce(fgt0, Fgt0 + (ntau + 1) * lnk, (ntau + 1) * lnk, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); // Sum the squares of them to calculate standard deviation
        
        
        MKL_Complex16 *FSIVCqt = NULL;
        MKL_Complex16 *FSIVCqt_allr = NULL;
        MKL_Complex16 *FSIVCqt_allr_pre = NULL;


        int total = (ntau+1) * n_dot * size; 
        if (rank == 0){
            FSIVCqt = new MKL_Complex16[(ntau+1)*n_dot*2];
            FSIVCqt_allr = new MKL_Complex16[(ntau+1)*n_dot*(size+1)];
            FSIVCqt_allr_pre = new MKL_Complex16[(ntau+1)*n_dot*(size+1)];

            for(int i=0;i<total;i++){
                FSIVCqt_allr_pre[i].real = 0.0;
                FSIVCqt_allr_pre[i].imag = 0.0;
            }

            std::string tmp;
            int read_rank, read_itau;
            double val;

            inf.open("FSIVCqt_allr.dat");
            if (inf.is_open()){
                for(int i_rank = 0; i_rank < size; i_rank++){
                    inf >> tmp >> read_rank;   
                    for(int itau = 0; itau < ntau+1; itau++){
                        inf >> tmp >> read_itau;

                        for(int iq = 0; iq < n_dot; iq++){
                            int Id0 = i_rank * (ntau+1) * n_dot + itau * n_dot + iq;
                            inf >> val;
                            FSIVCqt_allr_pre[Id0].real = val;
                        }
                    }
                }
            }
        }
        MPI_Gather(SIVCqt, (ntau + 1) * n_dot, MPI_C_DOUBLE_COMPLEX, FSIVCqt_allr, (ntau + 1) * n_dot, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); 

        

        if (rank == 0){
            for (int i = 0; i < size * (ntau+1) * n_dot; i++){
                FSIVCqt_allr[i].real = (FSIVCqt_allr_pre[i].real * measure_pre + FSIVCqt_allr[i].real) / (measure_pre + measure_times);
            }

            outf.open("q_points.dat");
            for(int q_i = 0; q_i < n_dot; q_i++){
                outf << q1[q_i] << " " << q2[q_i] << endl;
            }
            outf.close();

            outf.open("FSIVCqt_allr.dat");
            for(int i_rank = 0; i_rank < size; i_rank++){
                outf << "rank " << i_rank << endl;
                for(int itau = 0; itau < ntau+1; itau++){
                    outf<< "itau " << itau << endl;
                    for(int iq = 0; iq < n_dot; iq++){
                        Id0 = i_rank * (ntau+1) * n_dot + itau * n_dot + iq;
                        outf << FSIVCqt_allr[Id0].real << endl;
                    }
                }
            }
            outf.close();
        }



        MPI_Reduce(SIVCqt, FSIVCqt, (ntau + 1) * n_dot, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < (ntau+1) * n_dot; i++){
            SIVCqt[i].real = SIVCqt[i].real*SIVCqt[i].real;
            SIVCqt[i].imag = 0.0;
        }


        MPI_Reduce(SIVCqt, FSIVCqt + (ntau+1) * n_dot, (ntau+1) * n_dot, MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0){
            outf.open("SIVCqt.dat");
            outf << "kBT (meV)" << kBT / (1.0e-3 * e) << endl;
            outf << "q1, q2:" << endl;
            for(int q_i = 0; q_i < n_dot; q_i++){
                outf << q1[q_i] << " " << q2[q_i] << endl;
            }
            for(int I = 0; I < ntau+1; I++){
                outf<< "itau" << I << endl;
                for(int i = 0; i < n_dot; i++){
                    Id0 = I*n_dot + i;
                    FSIVCqt[Id0].real /= size;
                    val[0] = (FSIVCqt[(ntau + 1) * n_dot + Id0].real - pow(FSIVCqt[Id0].real, 2) * size) / size;
                    if(val[0] > 0.0)
                        FSIVCqt[(ntau + 1) * n_dot + Id0].real = sqrt(val[0]);
                    else
                        FSIVCqt[(ntau + 1) * n_dot + Id0].real = 0.0;
                    outf << FSIVCqt[Id0].real << " " << FSIVCqt[(ntau + 1) * n_dot + Id0].real << endl;

                }
            }
            outf.close();
        }

        if (rank == 0){
            outf.open("Covariance.dat");

            for(int i = 0; i < ntau+1; i++){
                for(int j = 0; j < ntau+1; j++){
                    outf << "i j " << i <<" "<< j << endl;
                    for(int q_i = 0; q_i < n_dot; q_i++){
                        MKL_Complex16 cijq = c0;
                        for(int bin = 0; bin < size; bin++){
                            int Id0 = i * (ntau+1) * n_dot + j * n_dot + q_i;
                            int Id1 = bin * (ntau + 1) * n_dot + i * n_dot + q_i;
                            int Id2 = i * n_dot + q_i;
                            int Id3 = bin * (ntau + 1) * n_dot + j * n_dot + q_i;
                            int Id4 = j * n_dot + q_i;

                            cijq.real += (FSIVCqt_allr[Id1].real - FSIVCqt[Id2].real) * (FSIVCqt_allr[Id3].real - FSIVCqt[Id4].real)-
                                        (FSIVCqt_allr[Id1].imag - FSIVCqt[Id2].imag) * (FSIVCqt_allr[Id3].imag - FSIVCqt[Id4].imag);
                            cijq.imag += (FSIVCqt_allr[Id1].real - FSIVCqt[Id2].real) * (FSIVCqt_allr[Id3].imag - FSIVCqt[Id4].imag) + 
                                        (FSIVCqt_allr[Id1].imag - FSIVCqt[Id2].imag) * (FSIVCqt_allr[Id3].real - FSIVCqt[Id4].real);
                        }
                        cijq.real /= size * (size - 1);
                        outf << cijq.real << endl;
                    }

                }
            }
            outf.close();
        }
        
        if (rank == 0) {
        
            outf.open("SACugt.dat", ios::app);
            outf << endl;
            outf << "average" << endl;
            for (int i = 0; i < lnk; i++) {
                outf << "momentum" << i << endl;
                for (int I = 0; I < ntau + 1; I++) {
                    Id0 = I * lnk + i;
                    Fgt0[Id0].real /= size;
                    outf << Fgt0[Id0].real << " ";
                }
                outf << endl;
                for (int I = 0; I < ntau + 1; I++) {
                    Id0 = I * lnk + i;
                    val[0] = (Fgt0[(ntau + 1) * lnk + Id0].real - pow(Fgt0[Id0].real, 2) * size) / size;
                    if (val[0] > 0.0)
                        Fgt0[(ntau + 1) * lnk + Id0].real = sqrt(val[0]); // Standard deviation as error
                    else
                        Fgt0[(ntau + 1) * lnk + Id0].real = 0.0;
                    outf << Fgt0[(ntau + 1) * lnk + Id0].real << " ";
                }
                outf << endl;
            }
            outf.close();
        }

        MKL_Complex16 *FCNut = NULL;
        if (rank == 0)
            FCNut = new MKL_Complex16[(size + 1) * (ntau + 1)];
        MPI_Gather(CNut, (ntau + 1), MPI_C_DOUBLE_COMPLEX, FCNut, (ntau + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); // Gather SIVC of all threads for SAC. One line in data.dat.
        if (rank == 0) {
            outf.open("SIVCut.dat", ios::app);
            for (int i = 0; i < size; i++) {
                outf << "rank" << i << endl;
                for (int j = 0; j < (ntau + 1); j++) {
                    Id0 = i * (ntau + 1) + j;
                    outf << std::setprecision(3) << FCNut[Id0].real << " ";
                }
                outf << endl;
            }
            outf.close();
        }
        MPI_Reduce(CNut, FCNut, (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int i = 0; i < (ntau + 1); i++) {
            CNut[i].real = CNut[i].real * CNut[i].real;
            CNut[i].imag = 0.0;
        }
        MPI_Reduce(CNut, FCNut + (ntau + 1), (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); // Sum the squares of them to calculate standard deviation
        if (rank == 0) {
            outf.open("SIVCutavg.dat", ios::app);
            for (int I = 0; I < ntau + 1; I++) {
                FCNut[I].real /= size;
                FCNut[I].imag /= size;
                outf << FCNut[I].real << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                outf << FCNut[I].imag << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                val[0] = (FCNut[(ntau + 1) + I].real - pow(FCNut[I].real, 2) * size) / size;
                if (val[0] > 0.0)
                    FCNut[I].real = sqrt(val[0]); // Standard deviation as error
                else
                    FCNut[I].real = 0.0;
                outf << FCNut[I].real << " ";
            }
            outf << endl;
            outf.close();
        }

        MKL_Complex16 *FCNut2 = NULL;
        if (rank == 0)
            FCNut2 = new MKL_Complex16[(size + 1) * (ntau + 1)];
        MPI_Gather(CNut2, (ntau + 1), MPI_C_DOUBLE_COMPLEX, FCNut2, (ntau + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); // Gather SIVC of all threads for SAC. One line in data.dat.
        if (rank == 0) {
            outf.open("SIVCut2.dat", ios::app);
            for (int i = 0; i < size; i++) {
                outf << "rank" << i << endl;
                for (int j = 0; j < (ntau + 1); j++) {
                    Id0 = i * (ntau + 1) + j;
                    outf << std::setprecision(3) << FCNut2[Id0].real << " ";
                }
                outf << endl;
            }
            outf.close();
        }
        MPI_Reduce(CNut2, FCNut2, (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int i = 0; i < (ntau + 1); i++) {
            CNut2[i].real = CNut2[i].real * CNut2[i].real;
            CNut2[i].imag = 0.0;
        }
        MPI_Reduce(CNut2, FCNut2 + (ntau + 1), (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); // Sum the squares of them to calculate standard deviation
        if (rank == 0) {
            outf.open("SIVCutavg2.dat", ios::app);
            for (int I = 0; I < ntau + 1; I++) {
                FCNut2[I].real /= size;
                FCNut2[I].imag /= size;
                outf << FCNut2[I].real << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                outf << FCNut2[I].imag << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                val[0] = (FCNut2[(ntau + 1) + I].real - pow(FCNut2[I].real, 2) * size) / size;
                if (val[0] > 0.0)
                    FCNut2[I].real = sqrt(val[0]); // Standard deviation as error
                else
                    FCNut2[I].real = 0.0;
                outf << FCNut2[I].real << " ";
            }
            outf << endl;
            outf.close();
        }

        MKL_Complex16 *FCNut3 = NULL;
        if (rank == 0)
            FCNut3 = new MKL_Complex16[(size + 1) * (ntau + 1)];
        MPI_Gather(CNut3, (ntau + 1), MPI_C_DOUBLE_COMPLEX, FCNut3, (ntau + 1), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); // Gather SIVC of all threads for SAC. One line in data.dat.
        if (rank == 0) {
            outf.open("SIVCut3.dat", ios::app);
            for (int i = 0; i < size; i++) {
                outf << "rank" << i << endl;
                for (int j = 0; j < (ntau + 1); j++) {
                    Id0 = i * (ntau + 1) + j;
                    outf << std::setprecision(3) << FCNut3[Id0].real << " ";
                }
                outf << endl;
            }
            outf.close();
        }
        MPI_Reduce(CNut3, FCNut3, (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int i = 0; i < (ntau + 1); i++) {
            CNut3[i].real = CNut3[i].real * CNut3[i].real;
            CNut3[i].imag = 0.0;
        }
        MPI_Reduce(CNut3, FCNut3 + (ntau + 1), (ntau + 1), MPI_C_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); // Sum the squares of them to calculate standard deviation
        if (rank == 0) {
            outf.open("SIVCutavg3.dat", ios::app);
            for (int I = 0; I < ntau + 1; I++) {
                FCNut3[I].real /= size;
                FCNut3[I].imag /= size;
                outf << FCNut3[I].real << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                outf << FCNut3[I].imag << " ";
            }
            outf << endl;
            for (int I = 0; I < ntau + 1; I++) {
                val[0] = (FCNut3[(ntau + 1) + I].real - pow(FCNut3[I].real, 2) * size) / size;
                if (val[0] > 0.0)
                    FCNut3[I].real = sqrt(val[0]); // Standard deviation as error
                else
                    FCNut3[I].real = 0.0;
                outf << FCNut3[I].real << " ";
            }
            outf << endl;
            outf.close();
        }

        MPI_Reduce(CN, FCN, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   // Sum over all threads
        MPI_Reduce(CN2, FCN2, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Sum over all threads
        CN[0] = CN[0] * CN[0];
        CN2[0] = CN2[0] * CN2[0];
        MPI_Reduce(CN, FCN + 2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   // Sum of squares to obtain standard deviation
        MPI_Reduce(CN2, FCN2 + 2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Sum of squares to obtain standard deviation

        if (rank == 0) {
            outf.open("gAF.dat", ios::out); // Output all auxiliary fields
            for (int i = 0; i < size * NumAF; i++)
                outf << gAF[i] << " ";
            outf.close();
            outf.open(filename.str(), ios::app);           // Output nonequal-time Green's function
            outf << "theta" << theta * 180.0 / pi << endl; // Temperature units in meV
            outf << "T=" << iT <<"meV"<< endl;
            FCN[0] /= size;
            FCN[1] /= size;
            FCN2[0] /= size;
            FCN2[1] /= size;
            FCN[2] = (FCN[2] - pow(FCN[0], 2) * size) / size;
            FCN2[2] = (FCN2[2] - pow(FCN2[0], 2) * size) / size;
            if (FCN[2] > 0.0)
                FCN[2] = sqrt(FCN[2]);
            else
                FCN[2] = 0.0;
            if (FCN2[2] > 0.0)
                FCN2[2] = sqrt(FCN2[2]);
            else
                FCN2[2] = 0.0;
            outf << "valley polarization" << endl;
            for (int i = 0; i < 3; i++)
                outf << FCN[i] << " ";
            outf << endl; // Change line
            outf << "intervalley coherance" << endl;
            for (int i = 0; i < 3; i++)
                outf << FCN2[i] << " ";
            outf << endl;
            outf.close();
        }

    } // temperature loop
    delete[] _Mmn;
    delete[] AF;
    delete[] fgt0;
    delete[] gAF;

    MPI_Finalize();
    return 0;
}
