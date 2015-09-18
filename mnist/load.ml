
(* #require "lacaml" *)

open Printf
open Bigarray
open Lacaml.D

let test_images_fname = ref "t10k-images-idx3-ubyte"
let test_labels_fname = ref "t10k-labels-idx1-ubyte"
let train_images_fname = ref "train-images-idx3-ubyte"
let train_labels_fname = ref "train-labels-idx1-ubyte"

let parse_labels fname =
  let ic = open_in_bin fname in
  let mn = input_binary_int ic in
  if mn <> 2049 then
    raise (Invalid_argument (sprintf "Wrong magic number got %d expected 2049" mn))
  else
    let size = input_binary_int ic in
    let data = Array.make size 0 in
    let rec loop i =
      try
        let () = Array.set data i (input_byte ic) in
        loop (i + 1)
      with End_of_file ->
        close_in ic;
        data
    in
    loop 0

let parse_images ?(normalize_by=Some 255.) fname =
  let ic = open_in_bin fname in
  let mn = input_binary_int ic in
  if mn <> 2051 then
    raise (Invalid_argument (sprintf "Wrong magic number got %d expected 2051" mn))
  else
    let size = input_binary_int ic in
    let rows = input_binary_int ic in
    let cols = input_binary_int ic in
    let feature_size = rows * cols in
    (*let () = Printf.printf "%d %d %d\n" size rows cols in *)
    (* For Onn each column is a training item *)
    let data = Mat.create feature_size size in
    let rec read_row r c =
      if c > feature_size then ()
      else
        let vb = float_of_int (int_of_char (input_char ic)) in
        let vn = match normalize_by with | None -> vb | Some m -> vb /. m in
        Array2.set data c r vn;
        read_row r (c + 1)
    in
    let rec loop i =
      try
        let () = read_row i 1 in
        loop (i + 1)
      with End_of_file ->
        close_in ic;
        data
    in
    loop 1

let row_to_square_gen r s =
  reshape (genarray_of_array1 r) [| s; s |]
  |> array2_of_genarray
  |> Mat.transpose

let join images labels =
  let m  = Mat.dim1 images in
  let n  = Mat.dim2 images in
  let td = Mat.make0 (m + 10) n in
  let td = lacpy ~b:td images in
  Array.iteri (fun idx label ->
    let row = label + 785 in (* 0 at 785 *)
    Array2.set td row (idx + 1) 1.)
    labels;
  td
