(* Load CIFAR-10 data
 * http://www.cs.toronto.edu/~kriz/cifar.html
 *)

open Printf

let train_batch1_fname = "data/cifar_10/cifar-10-batches-bin/data_batch_1.bin"
let train_batch1_fname = "data/cifar_10/cifar-10-batches-bin/data_batch_2.bin"
let train_batch1_fname = "data/cifar_10/cifar-10-batches-bin/data_batch_3.bin"
let train_batch1_fname = "data/cifar_10/cifar-10-batches-bin/data_batch_4.bin"
let train_batch1_fname = "data/cifar_10/cifar-10-batches-bin/data_batch_5.bin"
let test_batch_fname = "data/cifar_10/cifar-10-batches-bin/test_batch.bin"

let parse_file ~d ~t fname =
  let fd = Unix.openfile fname [Unix.O_RDONLY] 0o600 in
  let c = 10000 in
  let r = 3 * 32 * 32 in (* 3 for R, G, B *)
  let a = Array2.map_file fd Int8_unsigned C_layout false c (r + 1) in
  let labels = Array1.create d Fortran_layout c in
  let values = Array2.create d Fortran_layout c r in
  for i = 0 to c - 1 do
    labels.{i + 1} <- t (Array2.get a i 0);
    for j = 1 to r do
      values.{i + 1, j} <- t (Array2.get a i j)
    done;
  done;
  labels, values

let parse_file_raw = parse_file ~d:Int8_unsigned ~t:(fun x -> x)
let parse_file_float = parse_file ~d:Float64 ~t:float_of_int

let aligned_native f g =
  Array.init 32 (fun i ->
    Array.init 32 (fun j ->
      let o = 32 * i + j in
      g (f (o)) (f (o + 1024)) (f (o + 2048))))

let aligned_array2_fortran ~d f g =
  let a = Array2.create d Fortran_layout 32 32 in
  for i = 1 to 32 do
    for j = 1 to 32 do
      let o = 32 * i + j in
      Array2.unsafe_set a i j (g (f o) (f (o + 1024)) (f (o + 2048)))
    done
  done;
  a

let labels = function
  | 0 -> "airplane"
  | 1 -> "automobile"
  | 2 -> "bird"
  | 3 -> "cat"
  | 4 -> "deer"
  | 5 -> "dog"
  | 6 -> "frog"
  | 7 -> "horse"
  | 8 -> "ship"
  | 9 -> "truck"
  | x -> invalid_arg (sprintf "Only [0,9] acceptable CIFAR-10 labels: %d" x)
