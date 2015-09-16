
(*
#require "lacaml"
#require "ocplib-endian"
*)

open Printf
open Bigarray
open Lacaml.D
open EndianBytes

let test_images_fname = "t10k-images-idx3-ubyte"
let test_labels_fname = "t10k-labels-idx1-ubyte"
let train_images_fname = "train-images-idx3-ubyte"
let train_labels_fname = "train-labels-idx1-ubyte"

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

let parse_images fname =
  let ic = open_in_bin fname in
  let mn = input_binary_int ic in
  if mn <> 2051 then
    raise (Invalid_argument (sprintf "Wrong magic number got %d expected 2051" mn))
  else
    let size = input_binary_int ic in
    let rows = input_binary_int ic in
    let cols = input_binary_int ic in
    let feature_size = rows * cols in
    let () = Printf.printf "%d %d %d\n" size rows cols in
    let data = Mat.create size feature_size in
    let rec read_row r c =
      if c > feature_size then ()
      else
        let v = float_of_int (int_of_char (input_char ic)) in
        Array2.set data r c v;
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

