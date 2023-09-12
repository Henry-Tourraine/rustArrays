use crate::n_array::*;

pub trait Layer<T> where T:Number<T>{
    fn forward(&mut self, x: N_array<T>);
    fn backward(&mut self, output_error_gradient: N_array<T>, learning_rate: T);
    fn is_bias(&self)->bool;
    fn get_output(&self)->N_array<T> where T:Number<T>;
    fn get_input(&self)->N_array<T> where T:Number<T>;
}