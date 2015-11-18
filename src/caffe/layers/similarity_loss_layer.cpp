#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
//#if 0

namespace caffe {

template <typename Dtype>
void SimilarityLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  l1dists_.Reshape(bottom[0]->num(), 1, 1, 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i

  const int batch_size = bottom[0]->shape()[0];
  const int dim = bottom[0]->shape()[1];
  Dtype alpha = this->layer_param_.similarity_loss_param().alpha();
  Dtype beta = this->layer_param_.similarity_loss_param().beta();
  Dtype gamma = this->layer_param_.similarity_loss_param().gamma();
  
  Dtype loss(0.0);
  for (int i = 0; i < batch_size; ++i) {
    Dtype dw = caffe_cpu_asum(dim, diff_.cpu_data() + (i*dim));
    bool similar = (static_cast<int>(bottom[2]->cpu_data()[i]) == 0);
    l1dists_.mutable_cpu_data()[i] = dw;
    
    if (similar) {
      //alpha * dist_sq_.cpu_data()
      loss += alpha * dw * dw;
    } else {
      loss += beta * exp(gamma * dw);
    }
  }
  loss = loss / static_cast<Dtype>(batch_size);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void SimilarityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const int batch_size = bottom[0]->shape()[0];
  const int dim = bottom[0]->shape()[1];
  Dtype alpha = this->layer_param_.similarity_loss_param().alpha();
  Dtype beta = this->layer_param_.similarity_loss_param().beta();
  Dtype gamma = this->layer_param_.similarity_loss_param().gamma();
  Dtype *diff_signs = new Dtype[batch_size * dim];
  
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < dim; j++) {
      int pos = i*dim + j;
      diff_signs[pos] = (diff_.cpu_data()[pos] >= 0) ? 1 : -1; 
    }
  }

  for (int i = 0; i < 2; i++) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype coeff = sign * top[0]->cpu_diff()[0];
      for (int j = 0; j < batch_size; j++) {
        Dtype l1dist = l1dists_.cpu_data()[j];
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        bool similar = (static_cast<int>(bottom[2]->cpu_data()[j]) == 0);
        if (similar) {
          // alpha * 2 * Dw * (1/-1 for a/b) * sign(a_i - b_i)
          caffe_cpu_axpby(
              dim,
              alpha * 2 * coeff * l1dist,
              diff_signs + (j*dim),
              Dtype(0.0),
              bout + (j*dim));
        } else {
          // beta * gamma * l1dist * exp(gamma*l1dist) * sign(a_i - b_i)
          Dtype the_exp = exp(gamma * l1dist);
          caffe_cpu_axpby(
              dim,
              beta * gamma * coeff * l1dist * the_exp,
              diff_signs + (j*dim),
              Dtype(0.0),
              bout + (j*dim));
        }    
      }
    } 
  } 
  delete [] diff_signs;
}

#ifdef CPU_ONLY
STUB_GPU(SimilarityLossLayer);
#endif

INSTANTIATE_CLASS(SimilarityLossLayer);
REGISTER_LAYER_CLASS(SimilarityLoss);

}  // namespace caffe

//#endif
