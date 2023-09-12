use crate::n_array::*;
use crate::Tlayer::*;
//-----------------------------------------------------------------------------------
//CROSSCORRELATION

#[derive(Default)]
pub struct CrossCorrelation<T> where T: Number<T>{
    input_size: Vec<usize>,
    input: N_array<T>,

    input_gradients: N_array<T>,
    kernels: N_array<T>,
    kernels_gradients: N_array<T>,

    bias: N_array<T>,
    bias_gradients: N_array<T>,
    _bias: bool,
    
    pub output: N_array<T>,
    output_gradients: N_array<T>,
  
}

impl<T> CrossCorrelation<T> where T: Number<T>{
    fn rand(input: Vec<usize>, kernel_size: Vec<usize>, stride: usize, padding: usize, bias: bool)->Self{
        let output_shape: Vec<usize> = vec![
            kernel_size[kernel_size.len()-3],
            (2*padding+input[input.len()-2]-kernel_size[1])/stride,
            (2*padding+input[input.len()-1]-kernel_size[2])/stride,
            ];
        Self {
                input_size: input.clone(),
                _bias: bias,
                input: N_array::<T>{
                shape: input.clone(),
                data: Vec::<T>::with_capacity(input.clone().into_iter().reduce(|a, b| a*b).unwrap()).into_iter().map(|a| T::RANDOM()).collect()
                },
                input_gradients: N_array::<T>{
                    shape: input.clone(),
                    data: Vec::<T>::with_capacity(input.clone().into_iter().reduce(|a, b| a*b).unwrap()).into_iter().map(|a| T::RANDOM()).collect()
                },
                 kernels : N_array::<T>::rand(
                    kernel_size[..kernel_size.len()-2].to_vec().into_iter().chain((vec![input[0]]).into_iter()).chain(kernel_size[kernel_size.len()-2..].to_vec().into_iter()).collect::<Vec<usize>>()  
                  ),
                                                                      
                 kernels_gradients : N_array::<T>::rand(
                    kernel_size[..kernel_size.len()-2].to_vec().into_iter().chain((vec![input[0]]).into_iter()).chain(kernel_size[kernel_size.len()-2..].to_vec().into_iter()).collect::<Vec<usize>>()  
                  ),
                 
                 bias : N_array::<T>::rand(kernel_size.clone()),
                 bias_gradients : N_array::<T>::rand(kernel_size.clone()),
            
                output: N_array::<T>::rand(output_shape.clone()),
                output_gradients :  N_array::<T>::rand(output_shape.clone()),
            }
    }

    
}

impl<T>  Layer<T> for CrossCorrelation<T> where T: Number<T>{
    // input : c * h * w
    //                len 3                     len 3               len 1           len 2
    

    fn forward(&mut self, input: N_array<T>){
        if input.shape[0] != self.input_size[self.input_size.len()-3]{
            panic!("input {} and matrix {} not the same length ", input.shape[0], self.input_size[self.input_size.len()-3]);
        }

        let kernel_len = self.kernels.shape[1..].to_vec().into_iter().reduce(|a,b| a*b).unwrap();
        let bias_len = self.bias.shape[1..].to_vec().into_iter().reduce(|a,b| a*b).unwrap();

        let sub_kernel_len = self.kernels.shape[2..].to_vec().into_iter().reduce(|a,b| a*b).unwrap();
        let sub_bias_len = self.bias.shape[2..].to_vec().into_iter().reduce(|a,b| a*b).unwrap();

        let input_len = input.shape[1..].to_vec().into_iter().reduce(|a, b| a*b).unwrap();

        let output_len = self.output.shape[1..].to_vec().into_iter().reduce(|a, b| a*b).unwrap();
        let sub_output_len = self.output.shape[2..].to_vec().into_iter().reduce(|a, b| a*b).unwrap();
        println!("output length : {}", output_len);

        for d in 0..(self.output.shape.clone().into_iter().reduce(|a, b| a * b).unwrap()){
            self.output.data[d] = T::ZERO();
        }
        
        

        //through kernels length
        for k in 0..(self.kernels.shape[0]){
            let kernel = N_array::<T>{
                data: self.kernels.data[kernel_len * k..kernel_len * k + kernel_len].to_vec(),
                shape: self.kernels.shape[1..].to_vec()
            };

            let bias = N_array::<T>{
                data: self.bias.data[bias_len * k..bias_len * k + bias_len].to_vec(),
                shape: self.bias.shape[1..].to_vec()
            };

            

            //through input length
            for d in 0..self.input_size[0]{
                let sub_kernel = N_array::<T>{
                    data: kernel.data[sub_kernel_len * d..sub_kernel_len * d + sub_kernel_len].to_vec(),
                    shape: kernel.shape[1..].to_vec()
                };
                
                let input_ = N_array::<T>{
                    shape : input.shape[1..].to_vec(),
                    data : input.data[input_len * d..input_len * d + input_len].to_vec()
                };

                for y in 0..self.output.shape[self.output.shape.len()-2]{
                    for x in 0..self.output.shape[self.output.shape.len()-1]{
                        let mut input__ = input_.clone();
                        input__.crop(0 + x, 0 + y, sub_kernel.shape[sub_kernel.shape.len()-1], sub_kernel.shape[sub_kernel.shape.len()-2]);
                        let res = input__ * sub_kernel.clone() + bias.clone();
                        let sum = res.sum();
                        self.output.data[output_len * k + y * sub_output_len + x] += sum;
                        
                       

                    }
                }

            }
        }
        println!("{} {} ", self.output.shape.clone().into_iter().reduce(|a, b| a * b).unwrap(), self.output.data.len());
    }

    fn backward(&mut self, output_gradient: N_array<T>, learning_rate: T){
        for i in 0..self.kernels_gradients.data.len(){
            self.kernels_gradients.data[i] = T::ZERO();
        }
        for i in 0..self.input_gradients.data.len(){
            self.input_gradients.data[i] = T::ZERO();
        }
        let kernel_len = self.kernels_gradients.data.len()/self.kernels_gradients.shape[0];
        let input_len = self.input_gradients.data.len()/self.input_gradients.shape[0];
        //kernel len
        //input depth
        for i in 0..self.kernels.shape[0]{
            for j in 0..self.input_size[0]{
                //self.input[j], output_gradient[i], "valid"
                
                let mut res = (N_array::<T>::correlate2d(self.input[vec![j..j+1]].clone(), 
                                                    self.output_gradients[vec![i..i+1]].clone(),
                                                     1,
                                                     vec![0, 0]));

                //output_gradient[i], self.kernels[i, j], "full"
                let padding = self.kernels.shape[self.kernels.shape.len()-1];
                let mut res_ = N_array::<T>::convolve2d(output_gradient[vec![i..i+1]].clone(),
                                            self.output_gradients[vec![i..i+1, j..j+1]].clone(),
                                        1,
                                        vec![padding-1, padding-1]);

                for k in 0..res.data.len(){
                    self.kernels_gradients.data[kernel_len * i + input_len * j + k] = res.data[k];
                }
                for k in 0..res_.data.len(){
                    self.input_gradients.data[input_len * j + k] += (res[vec![k..k+1]]).data[0];
                }
            }
        }
        self.kernels -=  self.kernels_gradients.clone() * learning_rate;
        self.bias -=  output_gradient * learning_rate;
       
        self.output = self.input_gradients;
    }


    fn is_bias(&self)->bool{
        self._bias
    }


}

