use crate::n_array::*;
use crate::Tlayer::*;
//-----------------------------------------------------------------------------------
//ACTIVATION

pub trait Activation<T> where T: Number<T>{
    fn activation(&self, x:N_array<T>)->N_array<T>;
    fn activation_prime(&self, x:N_array<T>)->N_array<T>;

}

pub struct Tanh<T> where T:Number<T>{
    input: N_array<T>
}

impl<T> Activation<T> for Tanh<T> where T: Number<T>{
    fn activation(&self, x: N_array<T>)->N_array<T>{
        return x.tanh();
    }

    fn activation_prime(&self, x: N_array<T>)->N_array<T>{
        return  x.tanh().pow_i(2) * T::MINUS_ONE() + T::ONE(); 
    }
}

impl<T> Layer<T> for Tanh<T> where T: Number<T>{
    fn forward(&mut self, x: N_array<T>){
        self.input = x.clone();
        return self.activation(x);
        
    }
    fn backward(&mut self, output_error_gradient: N_array<T>, learning_rate: T){
        return output_error_gradient * self.activation_prime(self.input);
    }

    fn is_bias(&self){
        false;
    }
}


pub struct Sigmoid{}

impl<T> Activation<T> for Sigmoid where T: Number<T>{
    fn activation(&self, x: N_array<T>)->N_array<T>{
        return  ((x * T::MINUS_ONE()).exp() + T::ONE() ).inv();
    }

    fn activation_prime(&self, x: N_array<T>)->N_array<T>{
        let temp = self.activation(x);
        return temp.clone() * (  temp * T::MINUS_ONE() + T::ONE());
    }

    
}

impl<T> Layer<T> for Sigmoid where T: Number<T>{
    fn forward(&mut self, x: N_array<T>){
        self.input = x.clone();
        return self.activation(x);
        
    }
    fn backward(&mut self, output_error_gradient: N_array<T>, learning_rate: T){
        return output_error_gradient * self.activation_prime(self.input);
    }

    fn is_bias(&self){
        false;
    }
}

pub struct Softmax<T> where T: Number<T>{
    input: N_array<T>,
    output: N_array<T>,
    error: N_array<T>

}

impl<T> Activation<T> for Softmax<T> where T: Number<T>{

    fn activation(&self, x: N_array<T>)->N_array<T>{
        let temp = x.exp();
        return  temp.clone() / temp.sum();
    }

    fn activation_prime(&self, output_error_gradient: N_array<T>, x: N_array<T>)->N_array<T>{
        let soft = self.output.exp() /  self.output.exp().sum();
        let mut flat_soft = soft.clone();
        flat_soft.shape = vec!(1, soft.shape.clone().into_iter().reduce(|a, b| a * b).unwrap());
        let n = flat_soft.shape[1];
        
        return ((N_array::<T>::identity(n) - self.output.T()) * self.output.clone()) * output_error_gradient;
    }


}

impl<T> Layer<T> for Softmax<T> where T: Number<T>{
    fn forward(&mut self, x: N_array<T>){
        self.input = x.clone();
        return self.activation(x);
        
    }
    fn backward(&mut self, output_error_gradient: N_array<T>, learning_rate: T){
        return output_error_gradient * self.activation_prime(self.input);
    }

    fn is_bias(&self){
        false;
    }
}