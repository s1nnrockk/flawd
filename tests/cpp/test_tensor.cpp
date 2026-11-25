#include "flawd/tensor.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace {
void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void near(float a, float b, const char* message) { expect(std::fabs(a-b)<1e-5F,message); }
}
int main() {
 const auto a=flawd::Tensor::from_values({1,2,3});
 const auto b=flawd::Tensor::from_values({4,5,6});
 near(flawd::add(a,b)[0],5,"addition failed");
 near(flawd::subtract(b,a)[2],3,"subtraction failed");
 near(flawd::multiply(a,b)[1],10,"multiply failed");
 near(flawd::relu(flawd::Tensor::from_values({-2,3}))[0],0,"relu failed");
 const auto left=flawd::Tensor({2,2},{1,2,3,4});
 const auto right=flawd::Tensor({2,1},{5,6});
 const auto product=flawd::matmul(left,right);
 near(product[0],17,"matmul first value failed");
 near(product[1],39,"matmul second value failed");
 near(flawd::transpose_2d(left)[1],3,"transpose failed");
 const auto p=flawd::softmax(flawd::Tensor::from_values({1,2,3}));
 near(p[0]+p[1]+p[2],1,"softmax sum failed");
 bool rejected=false;
 try { (void)flawd::add(a,left); } catch(const std::invalid_argument&) { rejected=true; }
 expect(rejected,"shape mismatch was not rejected");
 std::cout<<"All FLAWD tensor tests passed.\n";
}
