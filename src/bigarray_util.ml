
let to_offset : type a. a layout -> (int -> int) = function
  | Fortran_layout -> (fun i -> i + 1)
  | C_layout       -> (fun i -> i)

(* TODO: make the 'f' in the to_array optional *)
module Array1 = struct
  include Array1
  
  let to_array ~f a = 
    let d = dim a in
    let o = to_offset (layout a) in
    Array.init d (fun i -> f (unsafe_get a (o i)))

end

module Array2 = struct
  include Array2
  
  let to_array ~f a = 
    let d1 = dim1 a in
    let d2 = dim2 a in
    let o = to_offset (layout a) in
    Array.init d1 (fun i ->
      Array.init d2 (fun j ->
        f (unsafe_get a (o i) (o j))))

end
