use crate::n_array::*;
use rand::Rng;
use std::collections::HashMap;
use std::io::Cursor;
use crate::n_array::type_of;




use crate::Tlayer::Layer;
use crate::loss::Loss;

pub fn hello(){
        println!("nn");
}



//-----------------------------------------------------------------------------------
//LINEAR

pub struct Linear<T> where T:Number<T>{
    inputs: N_array<T>,
    weights: N_array<T>,
    bias: Option<N_array<T>>,
    output: N_array<T>,
   


    weights_gradients: N_array<T>,
    bias_gradients: Option<N_array<T>>,
    input_gradients: N_array<T>,
}

impl<T> Linear<T> where T: Number<T>{
    fn new(input_shape:usize, output_shape: usize, bias: bool)->Self{
        let shape = vec![input_shape, output_shape];

        let temp = N_array::<T>::rand(shape.clone());
        let temp1 = N_array::<T>::rand(vec![shape[shape.len()-1]]);
        let temp2 = N_array::<T>::new(shape.clone(), T::default());

        return Self { 
            inputs: temp.clone(), 
            weights: temp.clone(), 
            bias: if bias {Some(temp1)} else{ None },
          
            weights_gradients: temp.clone(),
            bias_gradients: if bias {Some(temp.clone())} else{ None },
            output: temp.clone(),
            input_gradients: temp2 
        };
    }

    
}

impl<T> Layer<T> for Linear<T> where T: Number<T>{
    

    fn forward(&mut self, input: N_array<T>){
        if input.shape[input.shape.len()-1] != self.weights.shape[0]{
            panic!("input shape different from layer input shape");
        }
        self.output = (self.weights.clone() * input);
        
        match self.bias.clone(){
            Some(b)=>self.output += b,
            None=>println!("no bias"),
        }
    
    }

    fn backward(&mut self, output_error_gradient: N_array<T>, learning_rate: T){
        self.weights_gradients = output_error_gradient.clone() * self.inputs.clone().T();

        self.output = self.weights.clone().T() * output_error_gradient.clone();
        self.weights -=  self.weights_gradients.clone() * learning_rate;
        match self.bias.clone(){
            Some(b)=>self.bias = Some(b + self.bias_gradients.unwrap().clone() * learning_rate),
            None=>println!("no bias"),
        }
    }

    fn is_bias(&self)->bool{
        return match self.bias{
            Some(_)=>true,
            None=>false,
        }
    }

    fn get_output(&self)->N_array<T> where T:Number<T>{
        return self.output.clone();
    }

    fn get_input(&self)->N_array<T> where T:Number<T>{
        return self.inputs.clone();
    }
   
}



fn predict<T>(net: &Vec<Box<dyn Layer<T>>>, input: &N_array<T>) -> N_array<T> where T: Number<T>{
    let mut output = input.clone();
    for layer in net.iter(){
        layer.forward(output);
        output = layer.get_output();
    }
    return output;
}


fn train<T>(net: Vec<Box<dyn Layer<T>>>, loss: impl Loss<T>, x_train: Vec<N_array::<T>>, y_train: Vec<T>, epochs: u8, learning_rate: T, verbose: bool) where T: Number<T>{
    for epoch in 0..epochs{
        let error = T::ZERO();
        let t: Vec<(&N_array<T>, &T)> = x_train.iter().zip(y_train.iter()).collect();
        for (x, y) in t.iter(){
            let prediction = predict(&net, x.clone());
            let y_true = N_array::<T>{data: vec![*y.clone()], shape: vec![1]};

            error += Loss::f(
                y_true.clone(),
                prediction.clone()
            );

            let grad = Loss::f_prime(y_true.clone(), prediction.clone());

            for layer in net.into_iter().rev(){
                layer.backward(grad, learning_rate);
                grad = (layer.get_input());
            }
        }
     
            match type_of(error){
                "f64"=>error /= x_train.len() as f64,
                "f32"=>error /= x_train.len() as f32,
                _=>println!("None")
            }
        
        

        if verbose{
            println!("{}/{}, error={}", error+T::ONE(), epochs, error);
        }
    }
}



