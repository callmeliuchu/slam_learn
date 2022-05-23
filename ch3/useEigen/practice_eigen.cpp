#include <iostream>
using namespace std;
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main(){
    Eigen::Matrix<float,2,3> matrix_23;
    Eigen::Vector3d v_3d;
//    cout<<v_3d;
//    cout<<endl;
    Eigen::Matrix<float,3,1> vd_3d;
//    cout<<vd_3d<<endl;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
//    cout<<matrix_33<<endl;

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>matrix_dynamic;
//    matrix_dynamic(2,2) = 1.0;
//    cout<<matrix_dynamic<<endl;

    matrix_23<< 1,2,3,4,5,6;
    cout<<matrix_23<<endl;

    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            cout<<matrix_23(i,j)<<endl;
        }
    }
    cout<<"-----------"<<endl;
    v_3d << 3,2,4;
    cout<<v_3d<<endl;

    cout<<"-----------"<<endl;
    vd_3d << 4,5,6;
    cout<<vd_3d<<endl;
    cout<<"-----------"<<endl;


    Eigen::Matrix<double,2,1> result = matrix_23.cast<double>() * v_3d;
    cout<< result << endl;

    cout<<"--------"<<endl;

    Eigen::Matrix<float,2,1> result2 = matrix_23 * vd_3d;

    cout<< result2 << endl;

    cout<< "----------"<<endl;
    matrix_33 = Eigen::MatrixX3d::Random();
    cout<< matrix_33 << endl << endl;

    cout<< matrix_33.transpose() << endl;
    cout<< matrix_33.sum() << endl;
    cout<< matrix_33.trace() << endl;
    cout<< 10 * matrix_33 << endl;
    cout<< matrix_33.inverse() << endl;
    cout<< matrix_33.determinant() << endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    cout<< "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout<< "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;


    Eigen::Matrix<double,MATRIX_SIZE,MATRIX_SIZE> matrix_nn;
    matrix_nn = Eigen::MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    Eigen::Matrix<double,MATRIX_SIZE,1> v_nd;
    v_nd = Eigen::MatrixXd::Random(MATRIX_SIZE,1);

    clock_t start = clock();
    Eigen::Matrix<double,MATRIX_SIZE,1> x = matrix_nn.inverse() * v_nd;
    clock_t  end = clock();
//    cout<< x <<endl;
    cout<<"time spend1 "<< 1000* (end - start)/(double)CLOCKS_PER_SEC<<endl;
    start = clock();
    x = matrix_nn.colPivHouseholderQr().solve(v_nd);
    end = clock();
    cout<<"time spend2 "<< 1000* (end - start)/(double)CLOCKS_PER_SEC<<endl;
//    cout<<x<<endl;


}