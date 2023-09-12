use crate::n_array::*;
//-----------------------------------------------------------------------------------
//LOSS

pub trait Loss<T> where T: Number<T>{
    fn f(y_true: N_array<T>, y_pred: N_array<T>)->T;

    fn f_prime(y_true: N_array<T>, y_pred: N_array<T>)->N_array<T>;
}


pub struct MSE{}
impl<T> Loss<T> for MSE where T: Number<T>{
    fn f(y_true: N_array<T>, y_pred: N_array<T>)->T{
        (y_true - y_pred).pow_i(2).mean()
    }
    fn f_prime(y_true: N_array<T>, y_pred: N_array<T>)->N_array<T>{
        let mut len;
       
        ((y_true - y_pred) - T::ONE() - T::ONE()) / T::from(y_pred.data.len())
        
       
    }
}

pub struct Binary_cross_entropy{}
impl<T> Loss<T> for Binary_cross_entropy where T: Number<T>{
    fn f(y_true: N_array<T>, y_pred: N_array<T>)->T{
        ( y_true.clone() * T::MINUS_ONE() * y_pred.clone().log(T::ONE().exp())  
        - (( y_true * T::MINUS_ONE() + T::ONE()) 
        * (y_pred * T::MINUS_ONE() + T::ONE()).log(T::ONE().exp()))
        ).mean()
    }
    fn f_prime(y_true: N_array<T>, y_pred: N_array<T>)->N_array<T>{
        return match type_of(y_true.data[0]){
            "f64"=>(  y_true / y_pred  +   ( y_true * T::MINUS_ONE()  + T::ONE() ) * T::MINUS_ONE() / (y_pred * T::MINUS_ONE()  + T::ONE() ) ) / y_pred.data.len() as f64,
            "f32"=>(  y_true / y_pred  +   ( y_true * T::MINUS_ONE()  + T::ONE() ) * T::MINUS_ONE() / (y_pred * T::MINUS_ONE()  + T::ONE() ) ) / y_pred.data.len() as f32,
        }

    }
}
