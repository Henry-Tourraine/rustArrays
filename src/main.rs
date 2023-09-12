use std::cell::RefCell;
use std::ops::{Add, Sub, Mul, Div};
use std::rc::Rc;
use rand::Rng;
use std::fmt::{self, format};
use rustMath::nn::*;
use rustMath::n_array::*;

#[tokio::main]
fn main() {
    let X:N_array<f64> = N_array::<f64>::{
        data: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        shape: vec![4, 2, 1]
    };
    let Y:N_array<f64> = N_array::<f64>::{
        data: vec![0.0, 1.0, 1.0, 0.0],
        shape: vec![4, 1, 1]
    };
    println!("input {}", X.shape[0]);

    let sequential:Vec::<Box<dyn Layer>> = [
        Box::new(Linear::<f64>::new(
            X.shape[X.shape.len()-1],
            3
        )),
        Box::new(Linear::<f64>::new(
            3, 
            1
        ))

    ];
    



    

}
