use std::cell::RefCell;
use std::ops::{Add, Sub, Mul, Div, SubAssign, AddAssign, MulAssign, DivAssign};
use std::rc::Rc;
use rand::Rng;
use std::fmt;
use std::clone::Clone;
use std::sync::{Arc};
use std::ops::Range;
use std::f64::consts::PI;
use rand::prelude::Distribution;
use std::convert::TryFrom;
use std::any::type_name;


pub fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}


//<T as std::convert::TryFrom<usize>>::Error: Debug
pub trait Number<T>:  Add<Output = T> 
                    + AddAssign 
                    + Sub<Output = T> 
                    + SubAssign 
                    + Mul<Output = T>
                    + MulAssign
                    + Div<Output = T>
                    + DivAssign
                    + PartialEq
                    + Clone
                    + Default
                    + std::fmt::Display
                    + PartialOrd + Sized
                   
                    + From<usize>
                   // + TryFrom<usize>  // for unwrap
                     {
    fn RANDOM()->Self;

    fn ZERO()->Self;

    fn ONE()->Self;

    fn MINUS_ONE()->Self;

    fn INV(&self)->Self;

    fn PI()->Self;
    
    fn tanh(&self)->Self{
        return self.tanh();
    }
    fn powi(&self, i: i32)->Self{
        return self.powi(i);
    }
    fn powf(&self, f: f32)->Self{
        return self.powf(f);
    }
    fn exp(&self)->Self{
        return self.exp();
    }
    fn log(&self, base: T)->Self{
        return self.log(base);
    }
    fn cos(&self)->Self{
        return self.cos();
    }
    fn sin(&self)->Self{
        return self.sin();
    }
}




impl Number<f64> for f64{
    fn RANDOM()->f64{
        return rand::thread_rng().gen::<f64>();
    }
    fn ZERO()->Self{
        return 0.0;
    }
    fn ONE()->Self{
        return 1.0;
    }
    fn MINUS_ONE()->Self{
        return -1.0;
    }
    fn INV(&self)->Self{
        1.0 / self
    }

    fn PI()->Self{
        std::f64::consts::PI
    }
   
    
}
impl Number<f32> for f32{
    fn RANDOM()->f32{
        return rand::thread_rng().gen::<f32>();
    }
    fn ZERO()->Self{
        return 0.0;
    }
    fn ONE()->Self{
        return 1.0;
    }
    fn MINUS_ONE()->Self{
        return -1.0;
    }
    fn INV(&self)->Self{
        1.0 / self
    }
    fn PI()->Self{
        std::f32::consts::PI
    }
    
}

/*
impl Number<u8> for u8{
    fn random(&self)->u8{
        return rand::thread_rng().gen::<u8>();
    }
    fn ZERO()->Self{
        return 0;
    }
    fn ONE()->Self{
        return 1;
    }
    fn MINUS_ONE()->Self{
        return -1;
    }
    fn INV(&self)->Self{
        1 / self
    }
    fn PI()->Self{
        std::u8::consts::PI
    }
    
}
impl Number<u16> for u16{
    fn random(&self)->u16{
        return rand::thread_rng().gen::<u16>();
    }
    fn ZERO()->Self{
        return 0;
    }
    fn ONE()->Self{
        return 1;
    }
    fn MINUS_ONE()->Self{
        return -1;
    }
    fn INV(&self)->Self{
        1 / self
    }
    fn PI()->Self{
        std::u16::consts::PI
    }
    
}
*/



#[derive(Default, Clone)]
pub struct N_array<T: Number<T>>{
    pub shape: Vec<usize>,
    pub data: Vec<T>
}



impl<T> N_array<T> where T:Number<T>{
    pub fn new(shape: Vec<usize>, value:T) ->Self{
        let mut temp:Vec<T> = Vec::new();
        let mut length = 1;
        for i in 0..shape.len(){
            length = shape[i] * length;
        }

        for j in 0..length{
            temp.push(value.clone());
        }
        return Self{
            shape: shape,
            data: temp
        };
    }

    
    pub fn from_vec_iter(mut shape:Vec<usize>, dims: u8, data: Vec<T>)->Self{
        let mut temp:Vec<T> = Vec::new();
        let mut length = 1;

        shape = shape[..shape.len()-dims as usize].to_vec();

        let count = shape[shape.len() - dims as usize - 1..].to_vec();
        let mut count_prod = 1;
        for i in count.iter(){
            count_prod *= i;
        }
        println!("count len : {} count_prod : {}", count.len(), count_prod);

        if data.len()!= count_prod{
            panic!("vec size different from shape indicated. count_prod : {}, data : {}", count_prod, data.len());
        }

        fn fill<T:Number<T>>(shape: Vec<usize>, mut d:Vec<T>, data: Vec<T> )->Vec<T>{


            if shape.len()==0{
                let e = d.clone();
                for i in data.clone().into_iter(){
                    d.push(i);
                }
                
            }else{
                for i in 0..shape[0]{
                    d = fill(shape[1..].to_vec(), d, data.clone());
                }
            }

            

            return d;
        }

        let mut res:Vec<T> = Vec::new();
        res = fill(shape[0..shape.len()-dims as usize - 1].to_vec(),res, data);
        


        return Self{
            shape: shape,
            data: res
        };
    }

    pub fn rand(shape: Vec<usize>)->N_array<T>{
        let mut rng = rand::thread_rng();
        let mut temp: Vec<T> = Vec::new();


        for i in 0..shape.clone().into_iter().reduce(|a, b| a*b).unwrap(){
            temp.push(T::RANDOM());
        }

        return Self{
            shape: shape,
            data: temp
        }
    }

    pub fn flat_to_dims(&self, n: u8)->Vec<usize> {
        if n as usize >= self.shape.clone().to_vec().into_iter().reduce(|a, b| a*b).unwrap(){
            panic!("indice n out of range {} >= {}", n, self.shape.clone().to_vec().into_iter().reduce(|a, b| a*b).unwrap());
        }
        let mut indices :Vec<usize> = vec![];
        let mut rest = n;

        for i in 0..self.shape.len()-1{
            let s = self.shape[i];
            let m = self.shape.clone()[i+1..].to_vec().into_iter().reduce(|a, b| a*b).unwrap();
            let div = rest/m as u8;
            rest = rest%m as u8;
            indices.push(div as usize);
        }

        indices.push(rest as usize);

        indices
    }

    pub fn dims_to_flat(&self, dims: Vec<usize>)->usize{
        
        let mut indice = 0;
        let mut temp = self.data.clone();
        for i in 0..dims.len(){
            let length = temp.len()/self.shape[i];
            indice += length * dims[i];
            //println!("length {}, i {}", length, i);
            if i < dims.len()-1{
                temp = temp[length*i..length*i+length].to_vec();
            }
            

        }
        return indice as usize;
    }

    pub fn crop_multidims(&mut self, ranges: Vec<Range<usize>>)->N_array<T>{
       
        let data = self[ranges].clone();
        return self.clone();
    }

    pub fn T(&self)->Self{
        let rev_shape:Vec<usize> = self.shape.clone().into_iter().rev().collect::<Vec<usize>>();
        let buffer = self.shape.clone();
        let mut result = Self{
            shape: rev_shape,
            data: self.data.clone()
        };
        

        
        for i in 0..self.data.len(){

            let indices = self.flat_to_dims(i as u8);
            let rev_indices = indices.clone().into_iter().rev().collect::<Vec<usize>>();
            let v = self.data[i];
            
            let new_indice = result.dims_to_flat(rev_indices);
            result.data[new_indice] = v;
            
            
        }

        return result;
    
    }
    //WRONG
    pub fn get_from_indices(&self, shape: Vec<usize>)->T{

        let mut arr = self.data.clone();
        for i in 0..self.shape.len(){
            arr = arr[0 + shape[i]..(arr.len()/self.shape[i] + shape[i])].to_vec();

        }
        arr[0]
    }

    pub fn get_value_from_indices(&self, shape: Vec<usize>)->T{

        let mut arr = self.data.clone();

        
        for i in 0..self.shape.len(){
            let l = arr.len()/self.shape[i];
            
            arr = arr[shape[i] * l..(shape[i]) * l + l].to_vec();

        }
        arr[0]
    }

  
    pub fn tanh(&self)->Self{
        let mut temp = self.data.clone();
        for i in 0..temp.len(){
            temp[i] = temp[i].tanh();
        }
        return Self { shape: self.shape.clone(), data: temp }
    }

    pub fn pow_i(&self, n: i32)->Self{
        let mut temp = self.data.clone();
        for i in 0..temp.len(){
            temp[i] = temp[i].powi(n);
        }
        return Self { shape: self.shape.clone(), data: temp }
    }

    pub fn pow_f(&self, n: f32)->Self{
        let mut temp = self.data.clone();
        for i in 0..temp.len(){
            temp[i] = temp[i].powf(n);
        }
        return Self { shape: self.shape.clone(), data: temp }
    }

    pub fn exp(&self)->Self{
        let mut temp = self.data.clone();
        for i in 0..temp.len(){
            temp[i] = temp[i].exp();
        }
        return Self { shape: self.shape.clone(), data: temp }
    }

    pub fn log(&self, base: T)->Self{
        let mut temp = self.data.clone();
        for i in 0..temp.len(){
            temp[i] = temp[i].log(base);
        }
        return Self { shape: self.shape.clone(), data: temp }
    }

    pub fn sum(&self)->T{
        let mut temp: T;
        for i in self.data.iter(){
            temp +=  i.clone();
        }
        return temp
    }

    pub fn mean(&self)->T{
        let mut temp = self.sum();
        match type_of(temp){
            "f64"=>temp = temp / (self.data.len() as f64),
            "f32"=>temp = temp / (self.data.len() as f32),
            _=>println!("n_array -> mean error : type not covered"),
        }
       
        return temp;
    }

    pub fn identity(max: usize)->N_array<T>{
        let mut temp = N_array::<T>::default();
        temp.shape = vec![max, max];
        let index:usize = 0;
    
        for i in 0..max*max{
            if i%max==index{
                temp.data.push(T::ONE());
            }else{
                temp.data.push(T::ZERO());
            }
        }
        return temp
    }

    pub fn inv(&mut self)->&mut Self{
        for i in 0..self.data.len(){
            self.data[i] = self.data[i].INV();
        }
        return self;
    }


    pub fn add_padding(&mut self, value: T, mut padding: Vec<usize>){
        if self.shape.len()<2 {panic!("len must be 2")};
        let mut temp: Vec<T> = Vec::new();
        let mut indices:Vec<usize> = Vec::new();

        struct Loop_<'a, T> where T: Number<T>{
            f: &'a dyn Fn(&Loop_<T>, Arc<N_array<T>>, Vec<usize>, &mut Vec<T>, Vec<usize>, &mut Vec<usize>)
        }

        let l = Loop_::<T>{
            f: &|l, arr, shape, temp, indices, padding|{
        
        if arr.shape.len() == 2{

            for j in 0..padding[self.shape.len()-2]{
                for i in 0..arr.shape[arr.shape.len()-1]+padding[self.shape.len()-1]*2{
                    temp.push(value);
                }
            }
            
            for h in 0..arr.shape[arr.shape.len()-2]{
                
                    let mut  s = indices.clone();
                    s.push(h);
                    for i in 0..padding[self.shape.len()-1]{
                        temp.push(value);
                    }
                    for w in 0..arr.shape[arr.shape.len()-1]{
                        let mut m = s.clone();
                        m.push(w);

                        let mut  r = String::new();
                        for w in 0..m.len(){
                            r = format!("{} {}",r, w);
                        }
                        println!("check get value {}", r);
                        println!("check self {:?}", (arr.shape.iter().map(|a| format!("{} ", a)).collect::<Vec<String>>().join(""))   );
                        temp.push(arr.get_value_from_indices(m));
                    }
                    for i in 0..padding[self.shape.len()-1]{
                        temp.push(value);
                    }
            }
            for j in 0..padding[self.shape.len()-2]{
                for i in 0..arr.shape[arr.shape.len()-1]+padding[self.shape.len()-1]*2{
                    temp.push(value);
                }
            }
        }else{
            for i in 0..shape[0]{
                let mut ind = indices.clone();
                ind.push(i);
                (l.f)(l, Arc::new(self.clone()), shape[1..].to_vec(), temp, ind, padding);
            }

        }
        }
    };
    (l.f)(&l, Arc::new(self.clone()), self.shape.clone(), &mut temp, indices, &mut padding);

    let mut t: Vec<usize> = Vec::new();
    for i in 0..self.shape.len(){
        if i>=self.shape.len()-2{
            t.push(self.shape[i]+padding[self.shape.len()-i-1]*2);
        }else{
            t.push(self.shape[i]);
        }
    }


    println!("longueur {:?}", t.iter().map(|a| format!("{} ", a)).collect::<Vec<String>>().join(""));
    println!("data {}", temp.len());

    self.shape = t;
    self.data = temp;
   
    }

    pub fn remove_padding(&mut self, mut padding: Vec<usize>){
        if self.shape.len()<2 {panic!("len must be 2")};
        let mut temp: Vec<T> = Vec::new();
        let mut indices:Vec<usize> = Vec::new();

        struct Loop_<'a, T> where T: Number<T>{
            f: &'a dyn Fn(&Loop_<T>, Arc<N_array<T>>, Vec<usize>, &mut Vec<T>, Vec<usize>, &mut Vec<usize>)
        }

        let l = Loop_::<T>{
            f: &|l, arr, shape, temp, indices, padding|{
        
        if arr.shape.len() == 2{

            for h in 0..arr.shape[arr.shape.len()-2]{
                    if h < padding[0] || h>= arr.shape[arr.shape.len()-2]-padding[0]{
                        continue;
                    }
                    let mut  s = indices.clone();
                    s.push(h);
                    for w in 0..arr.shape[arr.shape.len()-1]{
                        if w < padding[1] || w >= arr.shape[arr.shape.len()-1]-padding[1]{
                            continue;
                        }
                        let mut m = s.clone();
                        m.push(w);
                   
                        println!("check indices {:?}", (m.iter().map(|a| format!("{} ", a)).collect::<Vec<String>>().join(""))   );
                        temp.push(arr.get_value_from_indices(m));
                    }
            }
        }else{
            for i in 0..shape[0]{
                let mut ind = indices.clone();
                ind.push(i);
                (l.f)(l, Arc::new(self.clone()), shape[1..].to_vec(), temp, ind, padding);
            }

        }
        }
    };
    (l.f)(&l, Arc::new(self.clone()), self.shape.clone(), &mut temp, indices, &mut padding);

    let mut t: Vec<usize> = Vec::new();
    for i in 0..self.shape.len(){
        if i>=self.shape.len()-2{
            t.push(self.shape[i]-padding[self.shape.len()-i-1]*2);
        }else{
            t.push(self.shape[i]);
        }
    }

    self.shape = t;
    self.data = temp;

    println!("longueur {:?}", (self.shape.iter().map(|a| format!("{} ", a)).collect::<Vec<String>>().join(""))   );
   
    }

    pub fn crop(&mut self, x: usize, y: usize, width: usize, height: usize){
        let mut a = Vec::<T>::new();
        let mut temp = Rc::new(RefCell::new(a));
        let mut temp_ = temp.clone();
        let s = Rc::new(RefCell::new(self));
        let s_ = s.clone();

        struct Loop_<'a>{
            f: &'a dyn Fn(&Loop_, Vec<usize>)
        };

        let l = Loop_{
            f: &move |l, shape: Vec<usize>|{
                
                if shape.len()==2{
                    for yy in 0..shape[0]{
                        for xx in 0..shape[1]{
                            if yy >= y && yy < y+height && xx >= x && xx < x+width{
                                temp_.borrow_mut().push(s_.clone().borrow().get_value_from_indices(vec![yy, xx]));
                            }
                        }
                    }

                }else{
                    for i in 0..shape[0]{
                        (l.f)(&l, shape[1..].to_vec());
                    }
                }
                
            }
        };
        (l.f)(&l, s.borrow().shape.clone());
        let mut inp = s.borrow().shape[0..s.borrow().shape.len()-2].to_vec();
        inp.extend(Vec::from([height, width]).iter());

        s.borrow_mut().shape = inp;
        s.borrow_mut().data = (temp.clone().borrow().to_vec());
    }
    

   pub fn correlate2d(mut m: N_array<T>, kernel: N_array<T>, stride: usize, padding: Vec<usize>)->N_array<T>{

      
        m.add_padding(T::ZERO(), padding);
        let mut output = N_array::<T>::new(vec![
            (m.shape[m.shape.len()-2] - kernel.shape[kernel.shape.len()-2]) / stride, 
            (m.shape[m.shape.len()-1] - kernel.shape[kernel.shape.len()-1]) / stride, 
                ], T::ZERO());

        for y in 0..output.shape[output.shape.len()-2]{
            for x in 0..output.shape[output.shape.len()-1]{
                let mut crop = m.clone();
                crop.crop(x, y, kernel.clone().shape[kernel.clone().shape.len()-1], kernel.clone().shape[kernel.clone().shape.len()-2]);
            let res = crop * kernel.clone();
                output.data[y * x + x] += res.sum();
            }
        }

        return output;
   }

   pub fn convolve2d(mut m: N_array<T>, kernel: N_array<T>, stride: usize, padding: Vec<usize>)->N_array<T>{

    m.add_padding(T::ZERO(), padding);
    let angle = 
    m = m * N_array::<T>{
        data: vec![T::PI().cos(), T::PI().sin() * T::MINUS_ONE(), T::PI().sin(), T::PI().cos()],
        shape: vec![2,2]
    };
    let mut output = N_array::<T>::new(vec![
        (m.shape[m.shape.len()-2] - kernel.clone().shape[kernel.clone().shape.len()-2]) / stride, 
        (m.shape[m.shape.len()-1] - kernel.clone().shape[kernel.clone().shape.len()-1]) / stride, 
            ], T::ZERO());

    for y in 0..output.shape[output.shape.len()-2]{
        for x in 0..output.shape[output.shape.len()-1]{
            let mut crop = m.clone();
            crop.crop(x, y, kernel.clone().shape[kernel.clone().shape.len()-1], kernel.clone().shape[kernel.clone().shape.len()-2]);
            let res = crop * kernel.clone();
            output.data[y * x + x] += res.sum();
        }
    }

    return output;
}



}


impl<T> std::ops::Index<Vec<Range<usize>>> for N_array<T> where T: Number<T>
{
    type Output = N_array<T>;
  
    fn index(&self, index: Vec<Range<usize>>) -> &Self::Output {
        
        return &self;
        
    }
}


impl<T> std::ops::IndexMut<Vec<Range<usize>>> for N_array<T> where T: Number<T>
{
    

    fn index_mut(&mut self, index: Vec<Range<usize>>) -> &mut N_array<T> {
        let mut r: Vec<T> = Vec::new();
        let len = self.shape[0];
        let len_ = self.shape[1];

        let mut start:usize = 0;
        let mut end:usize=0;
       

        struct Loop_<'a>{
            f: &'a dyn Fn(&Loop_, usize, Vec<usize>, Vec<Range<usize>>, usize, usize)->[usize; 2]
        }
        let l = Loop_{
            f: &|l,  length, temp_shape, ranges, start, end|->[usize; 2]{
                if temp_shape.len() == 0{ return [start, end]; };
                let length = length / temp_shape[0];
                println!("shape {}", length);
                let range = ranges[0].clone();
                let start_ = start+range.start*length;
                let end_ = start+range.end*length;
                return (l.f)(&l, length, temp_shape[1..].to_vec(), ranges[1..].to_vec(), start_, end_ );
            }
        };
        [start, end] = (l.f)(&l, self.data.len(), self.shape.clone(), index.clone(), start, end );


        println!("{} {}", start, end);
        self.data = self.data[start..end].to_vec();

        for i in 0..index.clone().len(){
            let range = &index.clone()[i];
            self.shape[i] = range.end - range.start;
        }
        
        return self;
        
    }
}


impl<T> fmt::Display for N_array<T> where T:Number<T>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        
        let mut temp = String::from("shape : ");
        for i in 0..self.shape.len(){
            temp += &format!("{}, ", self.shape[i]);
        }

       

        temp = temp[0..temp.len()].to_string();
        temp += "\ndata : \n[\n";
       

        fn print<T: Number<T>>(shape: Vec<usize>, indent: usize, mut data: Vec<T>, mut s: Rc<RefCell<String>>)->(Vec<T>, Rc<RefCell<String>>){
            let length = shape[shape.len()-1];

            let lf = |s:Rc<RefCell<String>>|->Rc<RefCell<String>>{
                for i in 0..indent-shape.len(){
                    s.borrow_mut().push_str(" ");
                }
                return s;
            };

            if shape.len()==1{
                s = lf(s);
                s.borrow_mut().push_str(&data[0..length].to_vec().into_iter().map(|a| format!("{}", a)).reduce(|a, b| a+", "+&b).unwrap());
                data = data[length..].to_vec();
                return (data, s);
            }
            let ss = s.clone();
           // s.borrow_mut().push_str("[");

            for i in 0..shape[0]{
                s = lf(s);
                ss.borrow_mut().push_str("[\n");
                s = lf(s);
                (data, s) = print(shape[1..].to_vec(), indent, data, s);
                s.borrow_mut().push_str("\n");
                s = lf(s);
                
                
                if i != shape[0]-1{
                    s.borrow_mut().push_str("],\n");
                }else{
                    s.borrow_mut().push_str("]\n");
                }
            }
            //s.borrow_mut().push_str("]\n");
            
            return (data, s);
        }
        let t = Rc::new(RefCell::new(temp));
        print::<T>(self.shape.clone(), self.shape.len()+1, self.data.clone(), t.clone());

        /*
        for i in 0..(self.data.len()/(self.shape[self.shape.len()-1])){
            for j in 0..(self.shape[self.shape.len()-1]){
                temp += &format!("{} ", self.data[i * self.shape[self.shape.len()-1] + j]);
            }
            temp+="\n";
            
        }
        */
        write!(f, "{}]", t.clone().borrow())


    }
}


impl<T> Add<N_array<T>> for N_array<T> where T: Number<T>{
    type Output = N_array<T>;

    fn add(self, mut other: N_array<T>) -> N_array<T>{

        let mut temp = Vec::new();

        match is_broadcastable(&self, &other){
            BROADCASTABLE::SAME_SAME=>(),
            BROADCASTABLE::SAME_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::SAME_DIFFERS=>(),
            BROADCASTABLE::DIF_SAME=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_DIFFERS=>(),
            BROADCASTABLE::M2_TOO_BIG=>()
    
        }
        for i in 0..other.data.len(){
            let t = self.data[i] + other.data[i];
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }
}

impl<T> Add<T> for N_array<T> where T: Number<T>{
    type Output = N_array<T>;

    fn add(self, other: T) -> N_array<T>{

        let mut temp = Vec::new();
        for i in 0..self.data.len(){
            let t = self.data[i] + other;
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }

}

impl<T> AddAssign<N_array<T>> for N_array<T> where T: Number<T>{
    fn add_assign(&mut self, other: Self) {
        //let a = self - other;
        *self = self.clone() + other;
    }
}

/*

impl<T> Add<N_array<T>> for T where T:Number<T>{
    type Output = N_array<T>;

    fn add(self, other: N_array<T>) -> N_array<T>{

        let mut temp:Vec<T> = Vec::new();
        for i in 0..other.data.len(){
            let t = self + other.data[i];
            temp.push(t);
        }


        N_array::<T>{
            shape: other.shape.clone(),
            data: temp
        }
    }

    
}
*/

impl<T> Mul for N_array<T> where T: Number<T>{

    type Output = N_array<T>;

    fn mul(self, mut other: N_array<T>) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();

        match is_broadcastable(&self, &other){
            BROADCASTABLE::SAME_SAME=>(),
            BROADCASTABLE::SAME_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::SAME_DIFFERS=>(),
            BROADCASTABLE::DIF_SAME=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_DIFFERS=>(),
            BROADCASTABLE::M2_TOO_BIG=>()
    
        }

        if other.shape.len()>=2 && self.shape.len()>=2{
           
            if self.shape[self.shape.len()-1] == other.shape[other.shape.len()-2]{
                println!("matrix multiplication");

             
                    let dims_sup: Vec<usize> = self.shape.clone()[0..self.shape.len()-2].to_vec();

                    fn loop_<T: Number<T>>(iter:Vec<usize>, indices_acc:Vec<usize>, m1: &N_array<T>, m2: &N_array<T>, temp: &mut Vec<T>){

                        if iter.len()==0{
                            for i in 0..m1.shape[m1.shape.len()-2]{
                                for j in 0..m2.shape[m2.shape.len()-1]{
            
                                    let mut t: T = T::default();
                                    let length = m1.shape[m1.shape.len()-1];
            
                                   
                                        for l in 0..length{
                            
                                            //println!("get from indices {} {}", self.get_from_indices(vec!(i, l)), other.get_from_indices(vec!(l, j)));
                                            t += m1.get_from_indices(indices_acc.clone().into_iter().chain(vec!(i, l).into_iter()).collect::<Vec<usize>>()) * m2.get_from_indices(indices_acc.clone().into_iter().chain(vec!(l, j).into_iter()).collect())
                                            
                                        }
                                    
                                   temp.push(t);
                                    
                                }
                            }

                            return;
                        }

                        for i in 0..iter[0]{
                            let mut new_indices_acc = indices_acc.clone();
                            new_indices_acc.push(i);
                            loop_(iter[1..iter.len()].to_vec(), new_indices_acc, m1, m2, temp);
                        }
                    }

                    loop_(dims_sup.clone(),  Vec::new(), &self, &other, &mut temp);

                    return N_array::<T>{
                        shape: dims_sup.into_iter().chain(Vec::from([self.shape[self.shape.len()-2], other.shape[other.shape.len()-1]]).into_iter()).collect(),
                        data: temp
                    }

            }else{
                if &self.shape.iter().zip(other.shape.iter()).filter(|&(a, b)| a==b).count() == &self.shape.clone().len(){
                    println!("element wise multiplication");
                    for i in 0..self.data.len(){
                            
                           temp.push(self.data[i] * other.data[i]);
                        
                    }

                    return N_array::<T>{shape: self.shape.clone(), data: temp}
                }
                panic!("A n : {} must match B m : {}", self.shape[self.shape.len()-2], other.shape[other.shape.len()-1]);

            }
        }else{
            panic!("matrices hasn't enough dimensions : dims A {}, dims B {}", self.shape.len(), other.shape.len());
        }

        


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }
}

impl<T> Mul<T> for N_array<T> where T: Number<T> {
    type Output = N_array<T>;

    fn mul(self, other: T) -> N_array<T>{

        let mut temp = Vec::new();
        for i in 0..self.data.len(){
            let t = self.data[i] * other;
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }

    
}



/*
impl<T> Mul<N_array<T>> for T where T: Number<T>{
    type Output = N_array<T>;

    fn mul(self, other: N_array<T>) -> N_array<T>{

        let mut temp = Vec::new();
        for i in 0..other.data.len(){
            let t = other.data[i] * self;
            temp.push(t);
        }


        N_array::<T>{
            shape: other.shape.clone(),
            data: temp
        }
    }

    
}
*/

impl<T> Sub<N_array<T>> for N_array<T> where T: Number<T>{
    type Output = N_array<T>;

    fn sub(self, mut other: N_array<T>) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();

        match is_broadcastable(&self, &other){
            BROADCASTABLE::SAME_SAME=>(),
            BROADCASTABLE::SAME_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::SAME_DIFFERS=>(),
            BROADCASTABLE::DIF_SAME=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_DIFFERS=>(),
            BROADCASTABLE::M2_TOO_BIG=>()
    
        }

        for i in 0..other.data.len(){
            let t = self.data[i] - other.data[i];
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }
}



impl<T> Sub<T> for N_array<T> where T: Number<T>{
    type Output = N_array<T>;

    fn sub(self, other: T) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();
        for i in 0..self.data.len(){
            let t = self.data[i] - other;
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }
}


/*
impl<T> Sub<N_array<T>> for T  where T: Number<T>{
    type Output = N_array<T>;

    fn sub(self, other: N_array<T>) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();
        for i in 0..other.data.len(){
            let t = self - other.data[i];
            temp.push(t);
        }


        N_array::<T>{
            shape: other.shape.clone(),
            data: temp
        }
    }
}
*/

impl<T> SubAssign<N_array<T>> for N_array<T>  where T: Number<T>{
    fn sub_assign(&mut self, other: Self) {
        //let a = self - other;
        *self = self.clone() - other;
    }
}


impl<T> Div<N_array<T>> for N_array<T>  where T: Number<T>{
    type Output = N_array<T>;

    fn div(self, mut other: N_array<T>) -> N_array<T>{

        match is_broadcastable(&self, &other){
            BROADCASTABLE::SAME_SAME=>(),
            BROADCASTABLE::SAME_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::SAME_DIFFERS=>(),
            BROADCASTABLE::DIF_SAME=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_ONE=>{
                other = broadcast(&self, &other);
            },
            BROADCASTABLE::DIF_DIFFERS=>(),
            BROADCASTABLE::M2_TOO_BIG=>()
    
        }

        let mut temp: Vec<T> = Vec::new();

        if self.shape.iter().zip(other.shape.iter()).filter(|(a,b)| **a==**b || **b == 1 as usize).count() == self.shape.len(){
            for i in 0..other.data.len(){
                let t = self.data[i] / other.data[i];
                temp.push(t);
            }

            N_array::<T>{
                shape: self.shape.clone(),
                data: temp
            }
    
        }else{

            //dimensions number is the same  1.values are the same  2.one or more values are 1 3. values differs


            //dimension number is not the same 1. values are the same 2.one or more values are 1 3. values differs
            panic!("not dividable");

        }
       
        

        
    }


}

impl<T> Div<T> for N_array<T>  where T: Number<T>{
    type Output = N_array<T>;

    fn div(self, other: T) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();
        for i in 0..self.data.len(){
            let t = self.data[i] / other;
            temp.push(t);
        }


        N_array::<T>{
            shape: self.shape.clone(),
            data: temp
        }
    }
}

/*
impl<T> Div<N_array<T>> for T  where T: Number<T>{
    type Output = N_array<T>;

    fn div(self, other: N_array<T>) -> N_array<T>{

        let mut temp: Vec<T> = Vec::new();
        for i in 0..other.data.len(){
            let t =  self / other.data[i];
            temp.push(t);
        }


        N_array::<T>{
            shape: other.shape.clone(),
            data: temp
        }
    }
}
*/

pub fn is_broadcastable<T: Number<T>>(m1: &N_array<T>, m2: &N_array<T>) ->BROADCASTABLE{
    
    if m1.shape.len() == m2.shape.len(){

        if m1.shape.iter().zip(m2.shape.iter()).filter(|(a, b)| a==b).count() == m1.shape.len(){
            println!("Same same");
            return BROADCASTABLE::SAME_SAME;

        }else if m1.shape.iter().zip(m2.shape.iter()).filter(|(a, b)| **a==**b || **b == 1 as usize).count() == m1.shape.len(){
            println!("Same one");
            return BROADCASTABLE::SAME_ONE;
        }else{
            println!("Same differs");
            return BROADCASTABLE::SAME_DIFFERS;
        }

    }else if m1.shape.len() > m2.shape.len(){
        println!("m1 > m2");
        m1.shape[m1.shape.len()-m2.shape.len()..m1.shape.len()].to_vec().iter().map(|x| {println!("value {}", x); return x}).collect::<Vec<&usize>>();

        if m1.shape[m1.shape.len()-m2.shape.len()..m1.shape.len()].to_vec().iter().zip(m2.shape.iter()).filter(|(a, b)| **a==**b).count() == m1.shape[m1.shape.len()-1-m2.shape.len()..m1.shape.len()-1].to_vec().len(){
            println!("Dif same");
            return BROADCASTABLE::DIF_SAME;

        }else if m1.shape[m1.shape.len()-m2.shape.len()..m1.shape.len()].to_vec().iter().zip(m2.shape.iter()).filter(|(a, b)| **a==**b || **b == 1).count() == m1.shape[m1.shape.len()-m2.shape.len()..m1.shape.len()].to_vec().len(){
            println!("dif one");
            return BROADCASTABLE::DIF_ONE;
        }else{
            println!("dif differs");
            return BROADCASTABLE::DIF_DIFFERS;
        }
    }else{
        println!("M2 too big");
        return BROADCASTABLE::M2_TOO_BIG;
    }
}


pub fn broadcast<T: Number<T>>(m1: &N_array<T>, m2: &N_array<T>)->N_array<T>{
    
    let mut lengths: Vec<usize> = Vec::new();
    let mut arr = m2.data.clone();
    let mut new_shape = m2.shape.clone();
    let mut new_data: Vec<T> = m2.data.clone();


    if new_shape.len() < m1.shape.len(){
        let mut temp = Vec::<usize>::new();
        for i in 0..(m1.shape.len()-m2.shape.len()){
            temp.push(1);
        }
        new_shape = temp.into_iter().chain(new_shape.into_iter()).collect();
    }

    let mut calc_lengths = move|new_data:&Vec<T>, new_shape:&Vec<usize>|->Vec<usize>{
        let mut temp:Vec<usize> = Vec::new();
        let mut d = new_data.clone();
        for i in new_shape.iter(){
            let length = d.len()/i;
            println!("length {}", length);
            temp.push(length);
            d = d[0..length].to_vec();
        }
    
        if temp.len() != new_shape.len(){
            panic!("length of lengths and shape differs");
        }

        return temp;
    };

 

    //For every dims
    for i in 0..m2.shape.len(){
        let dim2 = m2.shape[i];
        let dim1 = m1.shape[i];
        lengths = calc_lengths(&new_data, &new_shape);
        //case of broadcast
        if dim2 == 1 && dim1 != dim2{

            //get length of the chunk
            let length = lengths[i];
            //get number of times chunk must be added
            let expand_by = dim1;
            let mut temp: Vec<T> = Vec::new();

            //For every chunk of data at a given dim
            for j in 0..new_shape[..i].to_vec().into_iter().reduce(|a, b| a*b).unwrap(){
                let extract = new_data[j*length..length*(j+1)].to_vec();
                println!("check extract {}", length);
                for m in extract.iter(){
                    println!("{}", m);
                }
                
                for k in 0..expand_by{
                    temp = temp.into_iter().chain(extract.iter().cloned()).collect();
                }
            }

            new_data = temp;
            new_shape[i] = dim1;


        }else{
            println!("dim {} is {}", i, dim2);
        }

        
    }


    return N_array::<T>{
        shape: new_shape,
        data: new_data
    }
}


pub enum BROADCASTABLE{
    SAME_SAME, //SAME_SAME means same shape length and same values
    SAME_ONE, //SAME_ONE means same shape length but one dimension is one
    SAME_DIFFERS,// SAME_DIFFERS means same shape length but shape values differs
    DIF_SAME, // DIF_SAME means shape length differs but values are the same (ex : 4,5,7 and 5,7)
    DIF_ONE, // DIF_ONE means shape length differs and if values differs they are one (ex: 4,5,7 and 1,5 or 4,5,7 and 1,5,1)
    DIF_DIFFERS,// DIF_DIFFERS means shape length differs and values differs
    M2_TOO_BIG
}
